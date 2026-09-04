# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures, checkpoint access, and the package's single noise-floor definition.

Modelled on `models/demos/minimax_m3/tests/test_factory.py`. Four groups of helpers:

1. **Dimensions without a checkpoint** — `llama_config_dims()` reads the bundled
   `configs/Llama-3.1-8B-Instruct/config.json`, which is byte-identical to the staged checkpoint's
   (`bringup_log/00_MODEL_CARD.md` §1, `DEC-001`). Dimension-only tests need neither network nor
   weights.
2. **Checkpoint access** — `requires_hf_reference` skips when `HF_MODEL` is not a directory, and
   `load_hf_state_dict()` reads either the whole checkpoint or just the tensors a test asks for,
   straight from the safetensors shards (`DEC-012`).
3. **The noise floor** — `quantize_like_device()` and `err_ratio()`, copied from
   `models/demos/common/bringup/examples/noise_floor.py` as the package's ONE definition
   (`DEC-007`). Two copies drift, and then two gates disagree about what a floor is
   (recipe §2.2).
4. **Device-side setup** — `TestFactory.setup_test()` builds `MeshConfig` + `CCLManager` once per
   test. Added in P5.1, the phase that creates them (`DEC-008`); `ModelArgs` is not part of it until
   P6.2 (`DEC-012`), so the returned `hf` is the bundled config dict.
"""

import json
import os

import pytest

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links
from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
from models.demos.llama31_8b_d_p.tt.config import MeshConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_JSON = os.path.join(_HERE, "..", "configs", "Llama-3.1-8B-Instruct", "config.json")


def llama_config_dims() -> dict:
    """The bundled Llama-3.1-8B-Instruct `config.json`, verbatim (no HF, no network, no weights).

    This is also the dict the RoPE helpers must be fed: `get_rope_theta` /`get_rope_scaling`
    (`models/tt_transformers/tt/common.py:165`, `:183`) take a **dict**, and a live
    `transformers` config object has no `rope_theta` attribute at all — `getattr` with a default
    silently returns the default (recipe P1 trap 1, `07_RISKS.md` R-005).
    """
    with open(_CONFIG_JSON) as f:
        return json.load(f)


def bundled_config_path() -> str:
    """Path to the bundled `config.json` (used by the byte-identity assertion, `DEC-009`)."""
    return os.path.normpath(_CONFIG_JSON)


_HF_MODEL = os.getenv("HF_MODEL")

# Llama is first-class in `transformers` (no `trust_remote_code`), so the modeling code is always
# available; what this marker guards is the *weights*. Kept even though weights are staged on this
# machine, so the suite still runs on a weightless box (recipe §The machine).
requires_hf_reference = pytest.mark.skipif(
    not (_HF_MODEL and os.path.isdir(_HF_MODEL)),
    reason="set HF_MODEL to a Llama-3.1-8B-Instruct checkpoint directory to run HF-reference tests",
)


def hf_model_path():
    return _HF_MODEL


def load_hf_state_dict(prefixes=None, model_path=None) -> dict:
    """Load checkpoint tensors from the safetensors shards, optionally only those under `prefixes`.

    `prefixes` is an iterable of key prefixes (e.g. `("model.layers.0.",)`); `None` loads
    everything. Reading a subset is what keeps `G-REF` a host-only, few-second test instead of a
    15 GB load (`DEC-012`). Tensors come back at the checkpoint dtype (bf16); every caller casts to
    fp32 before computing anything, per the reference dtype policy (`DEC-006`,
    `bringup_log/01_REFERENCE.md` §3).
    """
    from safetensors.torch import load_file

    model_path = model_path or _HF_MODEL
    if not (model_path and os.path.isdir(model_path)):
        raise ValueError("HF_MODEL is not a directory; guard the caller with `requires_hf_reference`")

    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    wanted = {k: v for k, v in weight_map.items() if prefixes is None or any(k.startswith(p) for p in prefixes)}
    if not wanted:
        raise KeyError(f"no checkpoint keys match {prefixes}")

    state_dict = {}
    for shard in sorted(set(wanted.values())):
        shard_tensors = load_file(os.path.join(model_path, shard))
        state_dict.update({k: shard_tensors[k] for k in wanted if wanted[k] == shard})
    return state_dict


# --------------------------------------------------------------------------------------------
# Noise floor — the package's single definition (`DEC-007`; recipe §2.2 and
# `models/demos/common/bringup/examples/noise_floor.py`).
#
# Gate on the gap to the floor, never on another implementation's published PCC: its reference may
# share the device's own rounding and therefore report a flattered number (recipe §2.1). And the
# standing caveat (§2.3): a storage-dtype floor does NOT model a fused kernel's interior — SDPA
# alone measured 71x off its floor in the run this method comes from — so attribute a large ratio to
# a named stage before treating it as a bug.
# --------------------------------------------------------------------------------------------
def quantize_like_device(t, dtype):
    """Round `t` to exactly the values the device will hold, via ttnn, and return fp32.

    Host-only (no `device=` argument), so this is a pure quantiser and never a compute path.
    Reproduces `bfloat8_b`'s shared-exponent tile blocking exactly — which no hand-rolled torch
    emulation does. Requires a 4D, tile-shaped tensor for TILE_LAYOUT.
    """
    import ttnn

    assert t.dim() == 4, f"quantize_like_device expects a 4D tensor, got {tuple(t.shape)}"
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def err_ratio(measured: float, floor: float) -> float:
    """`(1 - measured) / (1 - floor)` — the measured error in units of the noise floor's.

    `1.0` means the module is exactly at the floor. `20x+` off the floor is a finding even when the
    absolute PCC looks pretty.
    """
    return float("inf") if floor >= 1.0 else (1.0 - float(measured)) / (1.0 - float(floor))


# --------------------------------------------------------------------------------------------
# Device-side setup (`DEC-008`; template `models/demos/minimax_m3/tests/test_factory.py:45`)
# --------------------------------------------------------------------------------------------
class TestFactory:
    """One place that builds the per-test device objects, so no test re-derives TP or `num_links`."""

    @staticmethod
    def setup_test(mesh_device, *, weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None, topology=None):
        """Build `MeshConfig` + `CCLManager` for an already-open `mesh_device`.

        TP is taken from the mesh's own column count, so a `(1,1)` P5 test gets TP=1 and the
        deployment `(4,8)` gets TP=8 — the only two shapes this package targets
        (`bringup_log/00_MODEL_CARD.md` §4). `num_links` comes from
        `models/demos/gpt_oss_d_p/utils/general_utils.py:27` (`DEC-013`).

        Unlike `models/demos/minimax_m3/tests/test_factory.py:56` this returns no `ModelArgs` and no
        `AutoConfig`: `ModelArgs` is a P6.2 deliverable (`DEC-012`), and the dims come from the
        bundled `config.json` with no HF import, which is what keeps every P5 unit test runnable on
        a bare card.
        """
        mesh_shape = tuple(mesh_device.shape)
        mesh_config = MeshConfig(mesh_shape, tp=mesh_shape[1])
        ccl_kwargs = {} if topology is None else {"topology": topology}
        ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), **ccl_kwargs)
        return {
            "mesh_device": mesh_device,
            "mesh_config": mesh_config,
            "ccl_manager": ccl_manager,
            "hf": llama_config_dims(),
            "weight_dtype": weight_dtype,
            "tensor_cache_path": tensor_cache_path,
        }
