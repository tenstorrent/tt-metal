"""vLLM lifecycle adapter for the production Nanbeige generator."""

import json
import os
import time
from pathlib import Path

import torch

# The package's tt_metal patches (patches.json nanbeige_4_3_33b-*) are opt-in switches read when a
# program is built. The device sampler must compile 64 candidates (top_k <= 64 and unset top-k run on
# device), so that switch is forced; the FP32 precision paths default on and TTNN_NANBEIGE_PRECISION=0
# turns them off.
os.environ["TTNN_NANBEIGE_SAMPLING_TOP64"] = "1"
os.environ.setdefault("TTNN_NANBEIGE_PRECISION", "1")

import ttnn

from .generator import Generator
from .precision import dtype as policy_dtype
from .precision import load_policy

HF_MODEL = "Nanbeige/Nanbeige4.3-33B"
HF_REVISION = "df919e1e2cf077a22204c78b6c826bf3db18bc08"


def mounted_snapshot():
    """tt-inference-server places the pinned checkpoint outside $HF_HOME and exports MODEL_WEIGHTS_DIR (or HF_MODEL)
    as a local snapshot directory. Prefer it when it holds config.json and the safetensors; a hub-cache snapshot
    directory must be the pinned revision."""
    for name in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        value = os.environ.get(name)
        if value and (Path(value) / "config.json").is_file() and any(Path(value).glob("*.safetensors")):
            resolved = Path(value).resolve()
            if resolved.parent.name == "snapshots" and resolved.name != HF_REVISION:
                raise ValueError(f"{name}={value} is snapshot {resolved.name}, not the pinned revision {HF_REVISION}")
            return Path(value)
    return None


def checkpoint_path(path=None):
    """The checkpoint directory: an explicit local path, then the mounted snapshot, then the pinned HF snapshot."""
    if path is not None and (Path(path) / "config.json").is_file():
        return Path(path)
    mounted = mounted_snapshot()
    if mounted is not None:
        return mounted
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_MODEL, revision=HF_REVISION, local_files_only=True))


class NanbeigeVLLM:
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_prefix_caching": True,
        "supports_sample_on_device": True,
        "supports_device_penalties": True,
        "supports_async_decode": True,
        "max_device_top_k": 64,
        "supports_device_top_k_unset": True,
        "supports_prefill_sampling_mask": True,
        "supports_original_prompt_lens": True,
        "reconcile_async_grammar_rows": True,
    }
    decode_input_update_contract = 1

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        *,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
    ):
        if tt_data_parallel != 1:
            raise ValueError("Nanbeige uses one TP4 mesh")
        self = cls()
        self.mesh = mesh_device
        self.config = hf_config
        self.slots = max_batch_size
        self.max_seq_len = max_seq_len
        self.tt_supported_decode_batch_sizes = [b for b in (1, 2, 4, 8, 16) if b <= max_batch_size]
        self.generator = None
        print(
            f"NANBEIGE_ADAPTER {__name__}:{cls.__name__} layers={hf_config.num_hidden_layers} slots={max_batch_size}",
            flush=True,
        )
        return self

    @staticmethod
    def get_max_tokens_all_users(*, max_num_seqs, **kwargs):
        # The plugin adds one output page per slot. Keep its final allocation
        # inside the measured stage-03 request pool, independent of slot count.
        return load_policy()["kv_pool_tokens"] - 32 * max_num_seqs

    def allocate_kv_cache(self, shape, dtype, num_layers):
        if tuple(shape[1:]) != (2, 32, 128) or dtype != torch.bfloat16:
            raise ValueError(f"Unexpected production KV specification {shape}, {dtype}")
        # vLLM's legacy interface has no torch dtype for TT block-float tiles.
        # Its BF16 spec describes logical values; the model callback owns physical
        # storage and get_max_tokens_all_users supplies the exact request-page cap.
        policy = load_policy()
        print(
            "NANBEIGE_KV_FORMAT",
            {
                "logical_vllm_dtype": str(dtype),
                "physical_ttnn_dtype": policy["kv_dtype"],
                "tile_bytes": 2048 if policy["kv_dtype"] == "bf16" else 1088,
                "request_pool_tokens": policy["kv_pool_tokens"],
                "allocated_pages": shape[0],
            },
            flush=True,
        )
        cache = [
            [
                ttnn.zeros(
                    shape,
                    device=self.mesh,
                    dtype=policy_dtype(policy["kv_dtype"]),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
            for _ in range(num_layers)
        ]
        self.generator = Generator(
            checkpoint_path(getattr(self.config, "_name_or_path", None)),
            self.mesh,
            max_batch_size=self.slots,
            max_seq_len=self.max_seq_len,
            num_layers=num_layers,
            kv_cache=cache,
            prepare=True,
        )
        if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1":
            import gc

            late = ttnn.empty([32, 32], device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            try:
                self.generator.step(1)
            except RuntimeError as error:
                if "unsafe" not in str(error).lower() and "allocation" not in str(error).lower():
                    raise
                print(
                    "TRACE_TRACKER_POSITIVE_CONTROL",
                    str(error).splitlines()[0],
                    flush=True,
                )
            else:
                raise RuntimeError("Trace allocation tracker failed to reject a live late allocation")
            del late
            gc.collect()
            self.generator.step(1)
        self._event(
            "prepared",
            programs=self.mesh.num_program_cache_entries(),
            layers=num_layers,
            memory=self.generator.model.memory_budget,
        )
        return cache

    def warmup_model_prefill(self, **kwargs):
        self.generator.prepare()

    warmup_model_decode = warmup_model_prefill

    def _sampling(self, kwargs):
        params = kwargs.get("sampling_params")
        if params is not None:
            from dataclasses import replace

            kwargs["sampling_params"] = replace(
                params,
                top_k=[-1 if k == self.config.vocab_size else k for k in params.top_k],
            )

    def _event(self, event, **values):
        if os.environ.get("NANBEIGE_TRACE_EVENTS"):
            print(
                "NANBEIGE_EVENT",
                json.dumps(
                    dict(
                        event=event,
                        time=time.time(),
                        stats=self.generator.stats,
                        traces={
                            b: [str(e.trace), str(e.sample_trace)] for b, e in self.generator.decode_states.items()
                        },
                        **values,
                    )
                ),
                flush=True,
            )

    def prefill_forward(self, *args, **kwargs):
        # The shared vllm-tt-plugin patch (supports_prefill_sampling_mask) passes the final-row mask as
        # prefill_sampling_mask; the generator takes it as sampling_mask.
        if "prefill_sampling_mask" in kwargs:
            kwargs["sampling_mask"] = torch.as_tensor(kwargs.pop("prefill_sampling_mask"), dtype=torch.bool)
        self._sampling(kwargs)
        out = self.generator.prefill_forward(*args, **kwargs)
        params = kwargs.get("sampling_params")
        self._event(
            "prefill",
            starts=[int(x) for x in kwargs["start_pos"]],
            ends=[int(x) for x in kwargs["prompt_lens"]],
            programs=self.mesh.num_program_cache_entries(),
            device_sampling=kwargs.get("sampling_params") is not None,
            base_seeds=(None if params is None else [None if seed is None else int(seed) for seed in params.seed]),
            sampled_tokens=out.reshape(-1).tolist() if params is not None else None,
        )
        return out

    def decode_forward(self, *args, **kwargs):
        self._sampling(kwargs)
        out = self.generator.decode_forward(*args, **kwargs)
        # Explicit diagnostic run only: capture the actual cached/mixed-bucket
        # path at a common-prefix decision for an independent FP32 comparison.
        # Disabled during allocation and performance measurements.
        capture_position = os.environ.get("NANBEIGE_CAPTURE_POSITION")
        if capture_position is not None:
            state = self.generator.decode_states[self.generator.current_bucket]
            positions = ttnn.to_torch(ttnn.get_device_tensors(state.positions)[0]).long().reshape(-1)
            if (positions == int(capture_position)).any():
                destination = Path(os.environ["NANBEIGE_CAPTURE_DIRECTORY"])
                destination.mkdir(parents=True, exist_ok=True)
                torch.save(
                    dict(
                        positions=positions,
                        logits=self.generator.read_logits(state.logits),
                        sampled_tokens=self.generator.read_tokens(self.generator.current_bucket, len(positions)),
                    ),
                    destination / f"decision_{time.time_ns()}.pt",
                )
        if kwargs.get("reload_inputs") or kwargs.get("reload_page_table"):
            params = kwargs.get("sampling_params")
            positions = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
            self._event(
                "decode_reload",
                bucket=self.generator.current_bucket,
                reload_inputs=kwargs.get("reload_inputs"),
                reload_page_table=kwargs.get("reload_page_table"),
                device_sampling=kwargs.get("sampling_params") is not None,
                reset_sampling_state=kwargs.get("reset_sampling_state"),
                positions=(None if positions is None else torch.as_tensor(positions).tolist()),
                base_seeds=(None if params is None else [None if seed is None else int(seed) for seed in params.seed]),
                slot_remap=(None if kwargs.get("slot_remap") is None else [int(i) for i in kwargs["slot_remap"]]),
            )
        return out

    def read_decode_output(self, *args, **kwargs):
        return self.generator.read_decode_output(*args, **kwargs)

    def process_decode_output_host(self, *args, **kwargs):
        return self.generator.process_decode_output_host(*args, **kwargs)

    def release_persistent_capture(self):
        if self.generator is not None:
            print("NANBEIGE_FINAL_STATS", json.dumps(self.generator.stats), flush=True)
            self.generator.close()
