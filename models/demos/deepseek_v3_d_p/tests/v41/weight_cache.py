# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The on-disk device-weight cache directory of V4.1 tests, keyed by what determines the stored bytes (bead 8y7.18.1).

Key = SOURCE + CONVERSION + mesh shape, nothing else:

* SOURCE: real weights -> the pinned checkpoint ``CHECKPOINT_REPO@CHECKPOINT_REVISION`` (``resolve_checkpoint``
  verifies the snapshot's index blob; the directory path is not identity); synthetic weights -> the seed, the
  weight-shaping model args and a digest of the synthetic-init source (``testing.py``).
* CONVERSION: the routed-expert dtype plus a digest of the conversion code only — the functions that turn
  weights into the stored device tensors (``CONVERSION_CODE``). Layout and mapper are fixed inside that code.
* mesh shape: device tensors are per-chip shards.

Excluded: the forward-pass sources (model.py / kernel_cpu.py / engram.py outside the conversion functions), so a
forward edit keeps the cache; the tt-metal git SHA (every commit would invalidate) and manual version constants
(a missed bump gives stale hits). Schedule-only args (layer lists, sequence length, top-k counts) are excluded too:
a layer's stored weights depend only on its id's role, which is asserted, so every schedule of the same weights
shares one directory (files are per layer). The oracle result cache (``oracle.cache_path``) is separate and
unchanged.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import asdict
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import kernel_cpu
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.oracle import OracleSpec
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig
from models.demos.deepseek_v3_d_p.tests.v41 import reference_weights
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
from models.demos.deepseek_v3_d_p.tt.v41 import weights as checkpoint_weights
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe

WEIGHT_CACHE = Path(os.environ.get("TT_V41_WEIGHT_CACHE", Path.home() / ".cache" / "tt-v41-weights"))

# The conversion boundary: everything between "weights as a torch tensor" (checkpoint shard or reference module)
# and the bytes written by ttnn.as_tensor / from_torch_and_dump. Function-level so that forward code in the same
# files (e.g. kernel_cpu's kernels next to unpack_fp4) stays out of the key.
CONVERSION_CODE = (
    # synthetic path: reference module -> device-layout torch weights
    reference_weights.dequant,
    reference_weights.device_weights,
    kernel_cpu.unpack_fp4,
    # real path: checkpoint shards -> device-layout torch weights
    checkpoint_weights.dequant_fp8_block,
    checkpoint_weights.dequant_mxfp4,
    checkpoint_weights.load_layer,
    checkpoint_weights.load_layer_dense,
    checkpoint_weights.load_routed_experts,
    checkpoint_weights._prepare,
    checkpoint_weights._dense_params,
    # both: torch weights -> stored device tensors (V4.1 bias recentring, shared-expert dtype; TtMoe caches)
    TtV41Moe.__init__,
    TtMoe.build_ttnn_cache,
    TtMoEGatePrefill._convert_and_cache_gate_weights,
    TtRoutedExpert._convert_and_cache_expert_weights,
    TtRoutedExpert.build_ttnn_cache,
    TtSharedExpert._convert_and_cache_weights,
    TtSharedExpert.build_ttnn_cache,
)
SYNTHETIC_INIT_SOURCE = Path(orc.__file__).parent / "testing.py"
# model args that describe the schedule or the runtime, not the stored weights of a layer (its role comes from its id)
SCHEDULE_ARGS = frozenset(
    {
        "max_batch_size",
        "max_seq_len",
        "temperature",
        "n_layers",
        "n_mtp_layers",
        "compress_ratios",
        "kv_source_layers",
        "index_source_layers",
        "candidate_source_layer",
        "candidate_topk_blocks",
        "index_topk",
        "engram_layer_ids",
        "engram_num_embeddings",
        "dspark_target_layer_ids",
    }
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:20]


def conversion_digest() -> str:
    return _sha("\n".join(inspect.getsource(f) for f in CONVERSION_CODE).encode())


def _check_roles(spec: OracleSpec) -> None:
    """Every layer of the spec has its V4.1 role (what makes per-layer files shareable across schedules)."""
    cfg = DeepSeekV41FlashConfig
    for pos, layer in enumerate(spec.layer_ids):
        assert spec.args.compress_ratios[pos] == cfg.compress_ratio(
            layer
        ), f"layer {layer} at position {pos} does not have its V4.1 role; the weight cache key assumes it does"


def source_key(spec: OracleSpec) -> str:
    if spec.checkpoint is not None:
        return f"hf:{checkpoint_weights.CHECKPOINT_REPO}@{checkpoint_weights.CHECKPOINT_REVISION}"
    shaping = {k: v for k, v in asdict(spec.args).items() if k not in SCHEDULE_ARGS}
    return "synthetic:" + orc._digest(spec.seed, shaping, _sha(SYNTHETIC_INIT_SOURCE.read_bytes()))


def weight_cache_dir(spec: OracleSpec, mesh_shape, routed_expert_weights_dtype=ttnn.bfloat8_b) -> Path:
    """The device-weight cache directory for the layers of ``spec`` on a mesh of ``mesh_shape`` (sp, tp)."""
    _check_roles(spec)
    sp, tp = tuple(mesh_shape)
    kind = "real" if spec.checkpoint is not None else "synthetic"
    key = orc._digest(source_key(spec), routed_expert_weights_dtype.name, conversion_digest(), [sp, tp])
    return WEIGHT_CACHE / f"{kind}-{key}-mesh{sp}x{tp}"
