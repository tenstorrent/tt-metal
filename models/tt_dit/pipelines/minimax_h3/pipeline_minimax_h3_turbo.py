# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 with a lightx2v Turbo distillation adapter fused into the transformer.

The adapter is a plain low-rank delta over the attention and feed-forward weights, so the generation
path is the base pipeline's: same packing, schedulers, VAE and audio decode. What changes is the
transformer's weights, the step count the caller asks for, and for the 768p files the video shift.
Those three travel together here rather than as switches on the base class.

A Turbo file publishes no sampling contract in its header. The caller supplies the step count, and
`num_inference_steps` counts sigma grid points, so a 4-forward adapter runs at 5 and an 8-forward one
at 9. The model card's shift for the file goes in `video_shift`; the 768p variants were distilled at
6 against the checkpoint's 12, and a wrong shift is a valid schedule over the wrong grid.

A HyperFlow file does publish a contract (see `hyperflow_minimax_h3`): its own sigma grid, and
interval `(t, r)` conditioning through a second, adapted time embedder. The step count then comes
from the file and any other is refused, except during warmup, whose output is discarded and which
keeps its short schedule. Both time embedders are float32 on device and the file ships their adapters
in float32, so they are fused on host rather than through the bfloat16 `register_lora` path.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn

from ...experimental.lora.h3_adapter_loader import H3AdapterHandle, h3_host_deltas, load_h3_adapter_into
from ...experimental.lora.promote import lora_modules
from ...models.transformers.minimax_h3.adaln_cache_minimax_h3 import MiniMaxH3AdalnCache
from ...models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3TimestepEmbedding, MiniMaxH3TwoTime
from .adaln_precompute import (
    AdalnTwoTime,
    MiniMaxH3AdalnLoraFold,
    MiniMaxH3AdalnTable,
    precompute_adaln_table,
    request_step_levels,
    slot_table_rows,
)
from .hyperflow_minimax_h3 import MiniMaxH3HyperFlow
from .packing import MINIMAX_H3_KEYFRAME_NOISE_AUG
from .pipeline_minimax_h3 import AUDIO_SHIFT, MINIMAX_H3_AUDIO_CONDITION_TIMESTEP, VIDEO_SHIFT, MiniMaxH3Pipeline
from .scheduler import MiniMaxH3Scheduler
from .weights_minimax_h3 import LORA_PATH_ENV, resolve_adapter_settings

#: Forward counts the published adapters were distilled for; `num_inference_steps` is one more.
TURBO_NUM_FORWARDS = (4, 8)

#: A HyperFlow file's float32 targets: the base time embedder and the endpoint copy it adds.
HYPERFLOW_HOST_PREFIXES = ("time_embedder.", "endpoint_time_embedder.")
_TIME_EMBEDDER_KEYS = ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")

#: Bumped whenever the table's row layout or build changes, so an old cached file is never read back.
_ADALN_TABLE_FORMAT = "levels-v2"


class MiniMaxH3TurboPipeline(MiniMaxH3Pipeline):
    """`MiniMaxH3Pipeline` whose transformer carries a lightx2v Turbo adapter.

    `lora_strength` multiplies the adapter's own `alpha / rank`, which the loader reads from the file;
    1.0 runs the adapter as trained.
    """

    def __init__(
        self,
        *,
        lora_path: str | os.PathLike,
        lora_strength: float = 1.0,
        **kwargs,
    ) -> None:
        # Bound onto the built transformer after the base weights load, never fused into the
        # checkpoint, so the weight cache stays adapter-independent and one cached copy serves every
        # adapter and strength.
        self.lora_path: Path = resolve_adapter_settings(
            lora_path=lora_path,
            lora_strength=lora_strength,
            default_video_shift=VIDEO_SHIFT,
            default_audio_shift=AUDIO_SHIFT,
        ).lora_path
        self.lora_strength = float(lora_strength)
        self._adapter: H3AdapterHandle | None = None
        self._time_embedder_states: dict[str, dict[str, torch.Tensor]] | None = None
        self._adaln_cache: MiniMaxH3AdalnCache | None = None
        self._adaln_table: MiniMaxH3AdalnTable | None = None
        self._adaln_slot_rows: dict[tuple[str, ...], list[torch.Tensor]] = {}
        self._lora_digest: str | None = None
        with safe_open(str(self.lora_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        video_shift, audio_shift = kwargs.get("video_shift"), kwargs.get("audio_shift")
        self.hyperflow = MiniMaxH3HyperFlow.from_adapter_metadata(
            metadata,
            video_shift=VIDEO_SHIFT if video_shift is None else float(video_shift),
            audio_shift=AUDIO_SHIFT if audio_shift is None else float(audio_shift),
        )
        super().__init__(**kwargs)

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        weights_dir: str | os.PathLike | None = None,
        lora_path: str | os.PathLike | None = None,
        lora_strength: float | None = None,
        video_shift: float | None = None,
        audio_shift: float | None = None,
        **kwargs,
    ) -> MiniMaxH3TurboPipeline:
        """`lora_path`, `lora_strength`, `video_shift` and `audio_shift` fall back to `MINIMAX_H3_LORA_PATH`,
        `MINIMAX_H3_LORA_STRENGTH`, `MINIMAX_H3_VIDEO_SHIFT` and `MINIMAX_H3_AUDIO_SHIFT`.

        Everything else is `MiniMaxH3Pipeline.create_pipeline`'s.
        """
        settings = resolve_adapter_settings(
            lora_path=lora_path,
            lora_strength=lora_strength,
            video_shift=video_shift,
            audio_shift=audio_shift,
            default_video_shift=VIDEO_SHIFT,
            default_audio_shift=AUDIO_SHIFT,
        )
        if settings.lora_path is None:
            raise ValueError(f"the Turbo pipeline needs an adapter: pass lora_path= or set {LORA_PATH_ENV}")
        return super().create_pipeline(
            mesh_device=mesh_device,
            weights_dir=weights_dir,
            video_shift=settings.video_shift,
            audio_shift=settings.audio_shift,
            lora_path=settings.lora_path,
            lora_strength=settings.lora_strength,
            **kwargs,
        )

    @property
    def adapter(self) -> H3AdapterHandle | None:
        """What is bound on the transformer, or None before its first residency."""
        return self._adapter

    def _prepare_transformer(self):
        transformer = super()._prepare_transformer()
        contract = self.hyperflow
        if self._adapter is None:
            if contract is not None:
                contract.assert_supports_task(self.task)
                contract.assert_supports_subfolder(self.transformer_subfolder)
            self._adapter = load_h3_adapter_into(
                transformer,
                str(self.lora_path),
                scale=self.lora_strength,
                name=self.lora_path.name,
                host_prefixes=HYPERFLOW_HOST_PREFIXES if contract is not None else (),
            )
            logger.info(
                f"turbo adapter {self._adapter.name}: {len(self._adapter)} targets, strength {self.lora_strength:g}"
            )
            # Under the precomputed AdaLN table both time embedders are folded into the table on host,
            # so the device transformer has no `time_embedder` to load or adapt.
            if contract is not None and not self._precomputed_adaln():
                self._time_embedder_states = self._fused_time_embedder_states()
                endpoint = MiniMaxH3TimestepEmbedding(
                    in_channels=transformer.time_embedder.linear_1.in_features,
                    hidden_dim=transformer.time_embedder.linear_1.out_features,
                    out_dim=transformer.time_embedder.linear_2.out_features,
                    mesh_device=self.mesh_device,
                )
                endpoint.load_torch_state_dict(self._time_embedder_states["endpoint_time_embedder"])
                transformer.two_time = MiniMaxH3TwoTime(embedder=endpoint, gate=contract.gate)
                transformer.time_embedder.load_torch_state_dict(self._time_embedder_states["time_embedder"])
                logger.info(
                    f"hyperflow: {contract.num_forwards} forwards, gate {contract.gate:g}, "
                    "both time embedders fused in float32"
                )
        else:
            # `coresident=False` evicts the transformer between stages and `cache.load_model` brings
            # back the cached *base* weights, so the fused delta has to be merged again. A no-op while
            # the weights stayed resident.
            for module in lora_modules(transformer):
                module.reapply_after_load()
            if self._time_embedder_states is not None and not self.coresident:
                transformer.time_embedder.load_torch_state_dict(self._time_embedder_states["time_embedder"])
        return transformer

    def _fused_time_embedder_states(self) -> dict[str, dict[str, torch.Tensor]]:
        """Base `time_embedder` weights plus each embedder's own float32 delta, as two state dicts."""
        base = self._read_checkpoint_tensors([f"time_embedder.{key}" for key in _TIME_EMBEDDER_KEYS])
        deltas = self._hyperflow_host_deltas()
        states = {}
        for prefix in HYPERFLOW_HOST_PREFIXES:
            state = {key: base[f"time_embedder.{key}"].float().clone() for key in _TIME_EMBEDDER_KEYS}
            for key in _TIME_EMBEDDER_KEYS:
                if key.endswith("weight"):
                    state[key] += deltas[f"{prefix}{key}"]
            states[prefix.rstrip(".")] = state
        return states

    def _hyperflow_host_deltas(self) -> dict[str, torch.Tensor]:
        """Both time embedders' float32 deltas, refusing a file that lacks any of them."""
        deltas = h3_host_deltas(str(self.lora_path), HYPERFLOW_HOST_PREFIXES, scale=self.lora_strength)
        expected = {
            f"{prefix}{key}"
            for prefix in HYPERFLOW_HOST_PREFIXES
            for key in _TIME_EMBEDDER_KEYS
            if key.endswith("weight")
        }
        if set(deltas) != expected:
            raise RuntimeError(f"hyperflow adapter time-embedder targets {sorted(deltas)}, expected {sorted(expected)}")
        return deltas

    def _read_checkpoint_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """A few named tensors from the transformer partition, without reading the other 62 GB."""
        directory = self.weights_dir / self.transformer_subfolder
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            shards = {key: directory / weight_map[key] for key in keys}
        else:
            shards = {key: directory / "diffusion_pytorch_model.safetensors" for key in keys}
        tensors = {}
        for key, shard in shards.items():
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                tensors[key] = handle.get_tensor(key)
        return tensors

    def _build_schedulers(self, num_inference_steps: int) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
        contract = self.hyperflow
        if contract is None or self._warming:
            return super()._build_schedulers(num_inference_steps)
        contract.assert_forwards(num_inference_steps)
        return self._contract_schedulers()

    def _contract_schedulers(self) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
        """Both modalities on the HyperFlow contract's own grid."""
        contract = self.hyperflow
        scheduler = MiniMaxH3Scheduler(shift=self.video_shift)
        audio_scheduler = MiniMaxH3Scheduler(shift=self.audio_shift)
        scheduler.set_timesteps(sigmas=contract.modality_sigmas(self.video_shift))
        audio_scheduler.set_timesteps(sigmas=contract.modality_sigmas(self.audio_shift))
        return scheduler, audio_scheduler

    def _step_endpoints(
        self, scheduler: MiniMaxH3Scheduler, audio_scheduler: MiniMaxH3Scheduler
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.hyperflow is None:
            return None
        return self.hyperflow.endpoints(scheduler.sigmas), self.hyperflow.endpoints(audio_scheduler.sigmas)

    # ------------------------------------------------------------------ precomputed AdaLN

    def _precomputed_adaln(self) -> bool:
        # A HyperFlow contract fixes the schedule before the first request, which is what makes the
        # table finite. A Turbo file leaves the step count to the caller, so it keeps the on-device
        # projections. MINIMAX_H3_ADALN_PRECOMPUTE=0 keeps them on device under HyperFlow too, for A/Bs.
        if os.environ.get("MINIMAX_H3_ADALN_PRECOMPUTE", "1") == "0":
            return False
        return self.hyperflow is not None

    def _prepare_adaln(self, slot_roles: tuple[str, ...]) -> tuple[MiniMaxH3AdalnCache, list[torch.Tensor]]:
        """The resident table for the contract's schedule and, per step, each slot's absolute row.

        Built (or loaded) and uploaded once per pipeline: the contract fixes the schedule, so warmup
        and serving share it. Warmup's own shorter schedule indexes this table modulo its length,
        which compiles the same programs; its output is discarded.
        """
        scheduler, audio_scheduler = self._contract_schedulers()
        contract = self.hyperflow
        video_endpoints = contract.endpoints(scheduler.sigmas)
        audio_endpoints = contract.endpoints(audio_scheduler.sigmas)
        audio_condition_timestep = MINIMAX_H3_AUDIO_CONDITION_TIMESTEP if self.task == "ref2va" else None
        if self._adaln_cache is None:
            step_levels = request_step_levels(
                scheduler.sigmas,
                audio_scheduler.sigmas,
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                audio_condition_timestep=audio_condition_timestep,
                video_endpoints=video_endpoints,
                audio_endpoints=audio_endpoints,
            )
            table = self._load_or_build_adaln_table(scheduler, audio_scheduler, step_levels)
            num_layers, hidden_size, _ = self._adaln_geometry()
            adaln_cache = MiniMaxH3AdalnCache(
                table,
                mesh_device=self.mesh_device,
                parallel_config=self.dit_parallel_config,
                num_layers=num_layers,
                hidden_size=hidden_size,
            )
            adaln_cache.assert_covers(len(step_levels))
            self._adaln_table, self._adaln_cache = table, adaln_cache
        if slot_roles not in self._adaln_slot_rows:
            self._adaln_slot_rows[slot_roles] = slot_table_rows(
                self._adaln_table,
                slot_roles,
                scheduler.sigmas,
                audio_scheduler.sigmas,
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                MINIMAX_H3_AUDIO_CONDITION_TIMESTEP,
                video_endpoints=video_endpoints,
                audio_endpoints=audio_endpoints,
            )
        return self._adaln_cache, self._adaln_slot_rows[slot_roles]

    def _adaln_geometry(self) -> tuple[int, int, int]:
        """`(num_layers, hidden_size, freq_dim)`, defaulting as the transformer does."""
        config = self.transformer_config
        return int(config.get("num_layers", 50)), int(config.get("hidden_size", 5376)), int(config.get("freq_dim", 256))

    def _load_or_build_adaln_table(
        self,
        scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
        step_levels: list[torch.Tensor],
    ) -> MiniMaxH3AdalnTable:
        path = self._adaln_cache_path(scheduler, audio_scheduler)
        # Unanimous, though both branches are host-only: a rank taking a cached early return while
        # another builds would leave the ranks on different tables, or skip the barrier below.
        if self._ranks_agree(path.is_file()):
            self._host_log(f"AdaLN table from cache: {path}")
            return torch.load(path, weights_only=False)

        self._host_log(
            f"building the AdaLN table on host for {len(step_levels)} forwards from "
            f"{self.transformer_subfolder}/ (reads the checkpoint)"
        )
        t0 = time.time()
        deltas = self._hyperflow_host_deltas()
        fold = MiniMaxH3AdalnLoraFold(
            {key: delta for key, delta in deltas.items() if not key.startswith(MiniMaxH3AdalnLoraFold.ENDPOINT_PREFIX)}
        )
        two_time = AdalnTwoTime(gate=self.hyperflow.gate, weight_hook=MiniMaxH3AdalnLoraFold.endpoint(deltas))
        num_layers, hidden_size, freq_dim = self._adaln_geometry()
        table = precompute_adaln_table(
            self.weights_dir / self.transformer_subfolder,
            step_levels,
            num_layers=num_layers,
            hidden_size=hidden_size,
            freq_dim=freq_dim,
            weight_hook=fold,
            two_time=two_time,
        )
        # Both folds: a spelling neither recognises still builds a valid table, from unadapted weights.
        for name, folded in (("base", fold), ("endpoint", two_time.weight_hook)):
            unapplied = folded.unapplied()
            if unapplied:
                raise KeyError(
                    f"{len(unapplied)} {name} AdaLN adapter target(s) matched no checkpoint key "
                    f"({', '.join(unapplied[:5])}); the table would be built from unadapted weights"
                )
        self._host_log(f"AdaLN table built in {time.time() - t0:.1f}s ({table.nbytes() / 1e9:.3f} GB); caching")

        distributed = ttnn.using_distributed_env()
        try:
            if not distributed or int(ttnn.distributed_context_get_rank()) == 0:
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(f".{os.getpid()}.tmp")
                torch.save(table, partial)
                partial.replace(path)
        except OSError as exc:
            # The table is already in memory, so a failed write costs a rebuild next run, on every
            # rank alike since `_ranks_agree` reads the same absent file.
            logger.warning(f"could not cache the AdaLN table to {path}: {exc}")
        finally:
            if distributed:
                ttnn.distributed_context_barrier()
        return table

    def _adaln_cache_path(self, scheduler: MiniMaxH3Scheduler, audio_scheduler: MiniMaxH3Scheduler) -> Path:
        """Disk location for a built table.

        The key covers everything the rows depend on, because a stale hit is silent: it modulates
        every block slightly wrong at every step, in the same direction.
        """
        root = os.environ.get("TT_DIT_CACHE_DIR")
        cache_dir = (Path(root) if root else Path.home() / ".cache" / "tt-dit") / "minimax-h3-adaln"
        grids = ";".join(
            ",".join(f"{sigma:.9g}" for sigma in schedule.sigmas.tolist()) for schedule in (scheduler, audio_scheduler)
        )
        key = "|".join(
            str(part)
            for part in (
                self.weights_dir.resolve(),
                self.transformer_subfolder,
                _ADALN_TABLE_FORMAT,
                f"sigmas=[{grids}]",
                self.hyperflow.identity(),
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                # ref2va carries a fourth level, the clean audio conditioning rows.
                self.task,
                *self._adaln_geometry(),
                self._lora_identity(),
            )
        )
        return cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.adaln.pt"

    def _lora_identity(self) -> str:
        """Content-hashed: adapters get overwritten in place, and a path key would read back a stale table."""
        if self._lora_digest is None:
            digest = hashlib.sha256()
            with self.lora_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(16 << 20), b""):
                    digest.update(chunk)
            self._lora_digest = digest.hexdigest()[:16]
        return f"lora={self._lora_digest}@{self.lora_strength:g}"

    @staticmethod
    def _ranks_agree(local: bool) -> bool:
        """Whether *every* rank sees `local` as true. A collective; all ranks must call it.

        A shared cache can disagree between hosts, and one rank skipping work the others do
        deadlocks rather than fails. Unanimity turns a partial cache into a rebuild.
        """
        if not ttnn.using_distributed_env():
            return local
        return all(ttnn.distributed_context_allgather_int(1 if local else 0))


__all__ = ["HYPERFLOW_HOST_PREFIXES", "TURBO_NUM_FORWARDS", "MiniMaxH3TurboPipeline"]
