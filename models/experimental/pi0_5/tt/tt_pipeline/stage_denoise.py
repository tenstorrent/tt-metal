# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""StageDenoise -- the drop-in adapter for the pi0.5 streamed-denoise port (plan §7).

Three entry points (plan §0 contract):
  * ``run_expert_chain(...)`` -- BYTE-FAITHFUL Galaxy parity: already-embedded suffix hidden
    in, raw post-expert hidden out, length-6 per-chip ``adarms_conds``, NO internal Euler
    loop. Correctness-only drop-in (eager, per-step re-precompute -- no trace).
  * ``run_denoise_loop(...)`` -- NEW loop-level driver: owns embed + N-step Euler + expert
    chain + final-norm + project + integrate as one streamed device trace (SC5 perf).
  * ``sample_actions(...)`` -- NEW full streamed path (thin wrapper over run_denoise_loop with
    an explicit prefix_kv_cache_torch, test-friendly).
Plus ``replay()`` (single-drain re-run of the captured loop) and ``close()``.

Topology (plan §7.0): explicit, non-inferential. parent_mesh=None -> Galaxy drop-in path
(denoise_submesh, 4-way only; n=8 raises). parent_mesh=<fresh mesh> -> standalone path
(4-way or 8-way). ZERO tt_symbiote imports.
"""
from __future__ import annotations

import warnings

import torch
import ttnn

from .denoise_block import TTNNPi05DenoiseExpertBlock
from .denoise_pipeline import (
    build_denoise_loop_pipeline,
    build_expert_only_pipeline,
    perf_suffix_len,
)
from ._d2d_pipeline import Pipeline
from .mesh_carve import carve_n_submeshes
from .weight_adapt import expert_reference_blocks, final_mod, suffix_reference

from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding  # noqa: F401 (parity import)

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

_L1 = ttnn.L1_MEMORY_CONFIG

_DEFAULT_SPLITS = {4: (5, 5, 4, 4), 8: (3, 3, 3, 3, 2, 2, 1, 1)}
_EXPERT_TOTAL_LAYERS = 18


def _resolve_topology(mesh_handles, parent_mesh, n_submeshes, splits):
    if parent_mesh is not None:  # PATH (b): standalone, caller owns the mesh
        parent = parent_mesh
        n = n_submeshes if n_submeshes is not None else 4
    else:  # PATH (a): Galaxy drop-in
        parent = mesh_handles.denoise_submesh  # (6,1)=6 chips
        n = n_submeshes if n_submeshes is not None else 4
        if n != 4:
            raise ValueError(
                "Galaxy drop-in path (parent_mesh=None) supports only n_submeshes=4: the "
                "denoise_submesh is (6,1)=6 chips. For 8-way, pass an explicit >=8-chip "
                "parent_mesh=... (standalone path). See plan §7.0."
            )
    if splits is None:
        splits = _DEFAULT_SPLITS.get(n)
        if splits is None:
            raise ValueError(f"no default split for n_submeshes={n}; pass splits=... summing to 18")
    assert sum(splits) == _EXPERT_TOTAL_LAYERS and len(splits) == n, (splits, n)
    return parent, n, splits


def _build_phantom_mask_and_offset(prefix_len, suffix_len, action_horizon):
    """Construct the suffix attention mask + scalar RoPE offset.

    iter-3 ROOT-CAUSE FIX (drill-down §6, SC1/SC2): the torch golden
    (``torch_paligemma.forward_expert``) runs the expert blocks with ``attention_mask=None`` --
    it does NOT mask the phantom pad-suffix keys ``[prefix_len+action_horizon :
    prefix_len+suffix_len]``. The iter-2 ``-1e4`` phantom band therefore made the DEVICE
    attention distribution diverge from the golden by ~11% rel-L2 PER LAYER, compounding over
    the 18 expert layers into a ~1.30x velocity-magnitude error. PCC (scale-invariant) hid it at
    0.991, but the Euler integrate ``x_t + dt*v`` is magnitude-sensitive -> the e2e collapsed to
    0.90. Single-block drill: (offset=prefix_len, NO mask) PCC 0.9999 vs (offset=0, WITH mask)
    PCC 0.9936 -- the MASK, not the RoPE offset, is the divergence. With the band removed, the
    full 18-layer velocity PCC rises 0.991->0.999 and the integrated PCC 0.902->0.997.

    The mask tensor is kept (all-zeros = a no-op additive mask) so the SDPA call site + trace
    shapes are unchanged; only the ``-1e4`` band is dropped to MATCH the golden. The RoPE offset
    (=prefix_len) is KEPT -- the (offset, NO mask) single-block drill confirmed it is correct
    (the prefix KV carries its own RoPE phase; relative positions are preserved). The final e2e
    slice ``[:, :action_horizon, :]`` already discards the pad rows.
    """
    mask = torch.zeros(1, 1, suffix_len, prefix_len + suffix_len)  # all-zeros = no masking (golden parity)
    position_offset = prefix_len  # RoPE slice base == prefix_len
    return mask, position_offset


def _remap_paired_to_flat(prefix_kv_per_chip, *, to_host=True):
    """Per-chip (K,V) -> flat 18-entry list; layer l = chip l//3, local l%3.

    to_host=True (default) materializes a torch list -- the original, topology-agnostic behaviour.
    to_host=False keeps the tensors on device, so the bind (_bind_prefix_kv) consumes them
    device-direct when co-resident on the consuming mesh; opt in only where co-residence holds."""
    if to_host:
        flat = [(ttnn.to_torch(k), ttnn.to_torch(v)) for chip in prefix_kv_per_chip for (k, v) in chip]
    else:
        flat = [(k, v) for chip in prefix_kv_per_chip for (k, v) in chip]
    assert len(flat) == _EXPERT_TOTAL_LAYERS, f"expected 18 flat KV layers, got {len(flat)}"
    return flat


class StageDenoise:
    """Drop-in denoise stage adapter. The positional signature
    ``(config, weights, mesh_handles, transport=None)`` is byte-identical to the Galaxy
    ctor; topology kwargs are keyword-only with Galaxy-faithful defaults."""

    def __init__(
        self, config, weights, mesh_handles, transport=None, *, parent_mesh=None, n_submeshes=None, splits=None
    ):
        ec = config.expert_config
        if ec.depth != _EXPERT_TOTAL_LAYERS:
            raise ValueError(f"expert depth must be {_EXPERT_TOTAL_LAYERS}, got {ec.depth}")
        self._config = config
        self._weights = weights
        self._suffix_cfg = SuffixConfig(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            expert_width=ec.width,
            pi05=True,
        )
        self._parent, self._n, self._splits = _resolve_topology(mesh_handles, parent_mesh, n_submeshes, splits)
        self._submeshes = carve_n_submeshes(self._parent, self._n)
        # Reference objects from the FULL Galaxy weights dict (plan §7.5).
        self._ref_blocks = expert_reference_blocks(weights["action_expert"], ec, depth=_EXPERT_TOTAL_LAYERS)
        self._final_w, self._final_b = final_mod(weights["action_expert"])
        self._ref_suffix = suffix_reference(weights["pi0_projections"], self._suffix_cfg)
        # transport accepted for Galaxy parity but IGNORED (the streamed pipeline owns its own
        # per-hop SplitSocketTransport; a single Galaxy SocketTransport cannot be split across
        # distinct submesh hops).
        self._transport_ignored = transport
        self._driver = None
        self._expert_pipe = None
        self._expert_kv_built = False

    # ---------------------------------------------------------------- helpers
    def _torch_adarms_cond(self, timestep):
        """Host-side adarms cond from a scalar/1d timestep via the reference suffix
        (sincos -> Linear -> silu -> Linear -> silu). Mirrors the device embed_adarms_cond."""
        t = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1)
        return self._ref_suffix.embed_timestep_adarms(t)

    # ---------------------------------------------------------------- NEW streamed entry points
    def run_denoise_loop(
        self,
        *,
        x_t_init_torch,
        num_steps,
        prefix_kv_per_chip,
        adarms_cond_per_step_torch=None,
        timesteps=None,
        prefix_len,
        action_horizon,
        attention_mask_torch=None,
        position_offset=None,
        capture=True,
    ):
        """Loop-level streamed denoise (SC5). Owns embed + N-step Euler + expert chain +
        final-norm + project + integrate as one streamed device trace."""
        prefix_kv_cache = _remap_paired_to_flat(prefix_kv_per_chip)
        return self._run_loop(
            x_t_init_torch=x_t_init_torch,
            num_steps=num_steps,
            prefix_kv_cache=prefix_kv_cache,
            adarms_cond_per_step_torch=adarms_cond_per_step_torch,
            timesteps=timesteps,
            prefix_len=prefix_len,
            action_horizon=action_horizon,
            attention_mask_torch=attention_mask_torch,
            position_offset=position_offset,
            capture=capture,
        )

    def sample_actions(
        self,
        *,
        x_t_init_torch,
        prefix_kv_cache_torch,
        num_steps,
        prefix_len,
        action_horizon,
        suffix_len=None,
        position_offset=None,
        attention_mask_torch=None,
        timesteps=None,
        adarms_cond_per_step_torch=None,
        drain="stage0",
        capture=True,
    ):
        """Full streamed path (thin wrapper) taking an explicit flat 18-entry
        prefix_kv_cache_torch (bypasses the Galaxy KV migrate). Test-friendly."""
        return self._run_loop(
            x_t_init_torch=x_t_init_torch,
            num_steps=num_steps,
            prefix_kv_cache=prefix_kv_cache_torch,
            adarms_cond_per_step_torch=adarms_cond_per_step_torch,
            timesteps=timesteps,
            prefix_len=prefix_len,
            action_horizon=action_horizon,
            attention_mask_torch=attention_mask_torch,
            position_offset=position_offset,
            suffix_len=suffix_len,
            drain=drain,
            capture=capture,
        )

    def _run_loop(
        self,
        *,
        x_t_init_torch,
        num_steps,
        prefix_kv_cache,
        adarms_cond_per_step_torch,
        timesteps,
        prefix_len,
        action_horizon,
        attention_mask_torch,
        position_offset,
        suffix_len=None,
        drain="stage0",
        capture=True,
    ):
        if suffix_len is None:
            suffix_len = perf_suffix_len(action_horizon)
        if adarms_cond_per_step_torch is None:
            if timesteps is None:
                ts, _ = _euler_timesteps(num_steps)
                timesteps = ts[:num_steps]  # one cond per step (the t at the START of each step)
            adarms_cond_per_step_torch = [self._torch_adarms_cond(t) for t in timesteps]
        assert len(adarms_cond_per_step_torch) == num_steps
        if attention_mask_torch is None or position_offset is None:
            mask, off = _build_phantom_mask_and_offset(prefix_len, suffix_len, action_horizon)
            if attention_mask_torch is None:
                attention_mask_torch = mask
            if position_offset is None:
                position_offset = off
        self._driver = build_denoise_loop_pipeline(
            self._ref_blocks,
            self._final_w,
            self._final_b,
            self._ref_suffix,
            self._config,
            self._suffix_cfg,
            self._parent,
            adarms_cond_per_step=adarms_cond_per_step_torch,
            prefix_kv_cache=prefix_kv_cache,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            attention_mask_torch=attention_mask_torch,
            position_offset=position_offset,
            num_steps=num_steps,
            action_horizon=action_horizon,
            splits=self._splits,
            submeshes=self._submeshes,
            block_cls=TTNNPi05DenoiseExpertBlock,
            use_concat_kv=True,
            drain=drain,
        )
        return self._driver.stream_euler(x_t_init_torch, capture=capture)[:, :action_horizon, :]

    def replay(self):
        assert self._driver is not None, "call run_denoise_loop/sample_actions(capture=True) first"
        return self._driver.replay()

    # ---------------------------------------------------------------- byte-faithful Galaxy parity
    def run_expert_chain(
        self,
        suffix_hidden_chip0,
        adarms_conds,
        prefix_kv_per_chip,
        attention_mask=None,
        position_ids=None,
        per_chip_attn_mask=None,
        per_chip_cos=None,
        per_chip_sin=None,
    ) -> "ttnn.Tensor":
        """BYTE-FAITHFUL Galaxy run_expert_chain: 18 expert layers over the port's own submeshes,
        input = already-embedded suffix hidden, output = RAW post-expert hidden on the last stage.
        NO embed, NO final-norm, NO project, NO Euler integrate. Eager, per-step re-precompute
        (no cache/replay -> no stale step-0 mods). Correctness-only drop-in (plan §7.6)."""
        if self._n != 6:
            raise ValueError(
                "run_expert_chain assumes the Galaxy 6-chip chip->stage identity (n==6). This port "
                f"carved n={self._n} submeshes; use run_denoise_loop for the streamed/perf path."
            )
        if not (isinstance(adarms_conds, (list, tuple)) and len(adarms_conds) == 6):
            raise ValueError("run_expert_chain expects a length-6 per-chip adarms_conds list")
        warnings.warn(
            "run_expert_chain is the correctness-only eager drop-in (re-precomputes mods per call, "
            "no trace). For the streamed perf win, migrate the caller to run_denoise_loop.",
            stacklevel=2,
        )
        # The 6 per-chip conds are mathematically identical replicas; read chip[0] to torch once.
        cond_torch = ttnn.to_torch(adarms_conds[0])
        prefix_kv_cache = _remap_paired_to_flat(prefix_kv_per_chip)
        prefix_len = prefix_kv_cache[0][0].shape[2]
        suffix_len = perf_suffix_len(self._config.action_horizon)
        # mask: prefer the per-chip / single mask the caller gave (read chip0 to torch); else build.
        if per_chip_attn_mask is not None:
            attn_mask_torch = ttnn.to_torch(per_chip_attn_mask[0])
            position_offset = prefix_len
        elif attention_mask is not None:
            attn_mask_torch = ttnn.to_torch(attention_mask)
            position_offset = prefix_len
        else:
            attn_mask_torch, position_offset = _build_phantom_mask_and_offset(
                prefix_len, suffix_len, self._config.action_horizon
            )

        if self._expert_pipe is None:
            # Build the bare expert pipeline ONCE; bind KV + mask but NOT mods (per-call re-precompute).
            self._expert_pipe = build_expert_only_pipeline(
                self._ref_blocks,
                self._ref_suffix,
                self._config,
                self._suffix_cfg,
                self._parent,
                adarms_cond_torch=cond_torch,
                prefix_kv_cache=prefix_kv_cache,
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                attention_mask_torch=attn_mask_torch,
                position_offset=position_offset,
                splits=self._splits,
                submeshes=self._submeshes,
                block_cls=TTNNPi05DenoiseExpertBlock,
                use_concat_kv=True,
                bind_mods=False,
            )
        # Re-precompute the per-step mods on each call (no stale step-0 mods).
        from .denoise_pipeline import _to_dram

        stages = self._expert_pipe.stages
        meshes = self._expert_pipe.meshes
        for k, st in enumerate(stages):
            cond_dev = ttnn.from_torch(cond_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=meshes[k])
            st._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in st.blocks]
            ttnn.deallocate(cond_dev)
        # Run the bare chain eagerly; input = the already-embedded suffix hidden.
        return self._expert_pipe(suffix_hidden_chip0)

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
        if self._expert_pipe is not None:
            try:
                self._expert_pipe.close()
            except Exception:
                pass
            self._expert_pipe = None
        # Release the inter-stage hop SocketTransports (tracked on the Pipeline class). driver/
        # expert_pipe close() above only drop their own loop traces + the wrap transport; the hop
        # sockets stay bound to the submeshes until released, which stalls the next multi-chip open.
        # The carved submeshes themselves are closed centrally by the test harness close_parent()
        # (one closer -> no double-close), mirroring the proven tt_symbiote teardown order:
        # release traces/transports (device still live) -> close submeshes -> close parent.
        try:
            Pipeline.release_all()
        except Exception:
            pass


def _euler_timesteps(num_steps):
    timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]
    dts = [timesteps[i + 1] - timesteps[i] for i in range(num_steps)]
    return timesteps, dts
