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
from contextlib import contextmanager

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


# --------------------------------------------------------------------------------------------
# P8: submeshes, and the fabric/topology pair that must be set together
#
# `BRINGUP_RECIPE.md:1708-1727`: on this galaxy a **top-level partial mesh** dies in fabric
# bring-up, because the routers on the opened devices wait for an ethernet handshake with
# partners outside the mesh. So every P8 shape below `(4, 8)` is a **submesh** of the full mesh
# (`tt_metal/api/tt-metalium/mesh_device.hpp:307`), measured by `G-FABRIC-MATRIX`.
# --------------------------------------------------------------------------------------------
GALAXY_MESH_SHAPE = (4, 8)

# `DEC-027` couples the fabric config and the collective topology behind ONE variable, because a
# Ring collective on a plain `FABRIC_1D` fabric **hangs** rather than erroring — and a hang on this
# box poisons every later collective until `tt-smi -r` (`G-FABRIC-MATRIX` case
# `submesh_1x8_ring_on_fabric1d`).
# **`Linear`, not `Ring`, on this galaxy** (`DEC-081`), and the reason is the ring SDPA rather than
# the plain collectives. `G-FABRIC-MATRIX` measured `ttnn.Topology.Ring` all-gathers returning
# bit-exact results on `FABRIC_1D` at every P8 shape — but
# `ttnn.transformer.ring_joint_scaled_dot_product_attention` under `Topology.Ring` asks the fabric
# for the SP axis's **wrap-around** route and aborts when it is missing:
#
#   TT_FATAL @ tt_metal/fabric/fabric.cpp:174: forwarding_direction.has_value()   (the condition is at `fabric.cpp:171`)
#   Could not find any forwarding direction from src (M0, D0) to dst (M0, D3)
#
# (D0 -> D3 is the 4-device SP ring closing on itself.) The same call with `Topology.Linear` runs and
# scores PCC 0.9996672 at 6.05x its own floor (`G-SP-RING`). So `Ring` is not merely unnecessary
# here, it is unserviceable for the one op that needs a real ring — which is a sharper statement than
# "the ring fabric will not initialise", and it is what settles the topology for the whole phase.
_TOPOLOGY_NAME = os.getenv("PREFILL_TOPOLOGY", "linear").lower()
_TORUS_DESCRIPTOR_BASENAME = "single_bh_galaxy_torus_xy_graph_descriptor.textproto"


def prefill_topology():
    """`ttnn.Topology` for this run: `Linear` by default on this machine, `Ring` via `PREFILL_TOPOLOGY`.

    The default is `linear` because this galaxy has no ring fabric for the SP ring SDPA (`DEC-079`,
    `DEC-097`, `R-030`) — an earlier draft of this docstring called `Ring` the "deployment default",
    which contradicted the `os.getenv` two lines above it (found by `G-CLEAN` item 5).
    """
    if _TOPOLOGY_NAME not in ("ring", "linear"):
        raise ValueError(f"PREFILL_TOPOLOGY must be 'ring' or 'linear', got {_TOPOLOGY_NAME!r}")
    return ttnn.Topology.Ring if _TOPOLOGY_NAME == "ring" else ttnn.Topology.Linear


# The fabric this galaxy actually has. **`FABRIC_1D`, even for Ring collectives** (`DEC-079`).
#
# `BRINGUP_RECIPE.md:80-90` says "The Ring topology P8 needs the torus descriptor; a Ring topology
# on a plain `FABRIC_1D` fabric **hangs** rather than erroring". `G-FABRIC-MATRIX` measured both
# halves of that on this box and **both are false here**:
#
#   * `FABRIC_1D_RING` cannot be initialised at all. The only single-galaxy RING/RING mesh-graph
#     descriptor is `single_bh_galaxy_torus_xy_graph_descriptor.textproto`, and it fails to map:
#     `topology_mapper.cpp:540` (the exception metal attributes to `:544`) — "Graph specified in MGD
#     could not fit in the discovered physical topology ... 32 target node(s) are not mapped to any
#     global node". It is not the channel
#     policy: a RELAXED copy fails identically, so the torus wrap links are not there. Asking
#     `FABRIC_1D_RING` of a LINE/LINE or LINE/RING descriptor instead is refused a step earlier, at
#     `mesh_graph.cpp:447-453` — "FabricConfig can only restrict topology (e.g., torus->mesh), not
#     create new connections". The two descriptors the recipe names are both multi-galaxy and out of
#     scope for a single galaxy (`DEC-071`).
#   * `ttnn.Topology.Ring` **collectives run correctly on `FABRIC_1D`**: bit-exact all-gathers at
#     `(1,8)`/1 link, `(2,8)`/2 links and `(4,8)`/2 links on **both** axes (`G-FABRIC-MATRIX`
#     addendum, 5/5).
#
# So the coupling `DEC-027` describes still holds — one variable decides both — but on this machine
# the pair is (`Topology.Ring`, `FABRIC_1D`). `PREFILL_FABRIC=1d_ring` is kept as an override for a
# genuinely torus-cabled machine, where the recipe's guidance would apply.
_FABRIC_NAME = os.getenv("PREFILL_FABRIC", "1d").lower()


def prefill_fabric_config():
    """The `ttnn.FabricConfig` that MUST accompany `prefill_topology()`. Never chosen separately."""
    if _FABRIC_NAME not in ("1d", "1d_ring"):
        raise ValueError(f"PREFILL_FABRIC must be '1d' or '1d_ring', got {_FABRIC_NAME!r}")
    return ttnn.FabricConfig.FABRIC_1D_RING if _FABRIC_NAME == "1d_ring" else ttnn.FabricConfig.FABRIC_1D


def galaxy_device_params():
    """`device_params` for a P8 test: the fabric config matching `PREFILL_TOPOLOGY`.

    Use as `@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)`
    (pattern: `models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py:126`).
    """
    return {"fabric_config": prefill_fabric_config()}


# A `FABRIC_1D_RING` run additionally needs the **torus** mesh-graph descriptor, and that one cannot
# be set from inside pytest: metal reads `TT_MESH_GRAPH_DESC_PATH` when the control plane
# initialises. On this galaxy that descriptor does not map (see `_FABRIC_NAME` above), so
# `PREFILL_FABRIC` defaults to `1d` and this marker skips only the opt-in case — rather than
# silently running a fabric init that aborts (`DEC-076`).
def _torus_descriptor_is_set() -> bool:
    path = os.getenv("TT_MESH_GRAPH_DESC_PATH", "")
    return bool(path) and os.path.basename(path) == _TORUS_DESCRIPTOR_BASENAME and os.path.isfile(path)


requires_ring_fabric = pytest.mark.skipif(
    _FABRIC_NAME == "1d_ring" and not _torus_descriptor_is_set(),
    reason=(
        f"PREFILL_FABRIC=1d_ring needs the single-galaxy torus descriptor exported before pytest "
        f"starts: TT_MESH_GRAPH_DESC_PATH="
        f"$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/{_TORUS_DESCRIPTOR_BASENAME}. "
        f"On this galaxy that descriptor does not map to the physical topology at all "
        f"(G-FABRIC-MATRIX); the default PREFILL_FABRIC=1d is what runs here."
    ),
)

requires_galaxy = pytest.mark.skipif(
    ttnn.get_num_devices() < GALAXY_MESH_SHAPE[0] * GALAXY_MESH_SHAPE[1],
    reason=f"P8 needs the full {GALAXY_MESH_SHAPE} Blackhole Galaxy ({GALAXY_MESH_SHAPE[0] * GALAXY_MESH_SHAPE[1]} devices)",
)


class SubmeshPool:
    """Hands out submeshes of one open parent mesh, with `quiesce_devices()` **enforced**.

    `tt_metal/api/tt-metalium/mesh_device.hpp:296-305` requires a barrier "between phases that use
    overlapping submeshes on the same physical devices" and names `quiesce_devices()`.
    **Nothing enforces it**, and forgetting it does not fail — it hangs the machine, and the hang is
    not contained: every later collective on the box hangs too, including ones that just passed,
    until `tt-smi -r` (`BRINGUP_RECIPE.md:1740-1745`, `LANDMINES.md`, and `G-FABRIC-MATRIX` case
    `overlap_1x2_then_1x8_no_quiesce` measures it).

    So this pool quiesces on **both** sides of every hand-out (`DEC-077`): a barrier before the
    phase starts and one after it ends. That makes "two live submeshes with no barrier between
    their phases" unreachable through this API, which is stronger than remembering the call — and
    `G-TP-PARITY`, which compares `(1,1)` against five multi-device shapes in one process, is
    exactly the pair the header warns about.

    Submeshes are **cached** by `(shape, offset)` and reused, so a five-shape sweep creates five
    submeshes rather than one per parametrisation.
    """

    def __init__(self, parent):
        self.parent = parent
        self._cache = {}

    @contextmanager
    def use(self, shape, offset=None):
        """Yield a submesh of `shape`, with a barrier before and after the phase."""
        shape = tuple(shape)
        key = (shape, tuple(offset) if offset is not None else None)
        self.parent.quiesce_devices()
        if key not in self._cache:
            mesh_offset = ttnn.MeshCoordinate(*offset) if offset is not None else None
            self._cache[key] = self.parent.create_submesh(ttnn.MeshShape(*shape), mesh_offset)
        sub = self._cache[key]
        try:
            yield sub
        finally:
            ttnn.synchronize_device(sub)
            self.parent.quiesce_devices()


@pytest.fixture(scope="function")
def submesh_pool(mesh_device):
    """A `SubmeshPool` over the full galaxy mesh. Every P8 shape below `(4,8)` comes from here."""
    assert tuple(mesh_device.shape) == GALAXY_MESH_SHAPE, (
        f"the submesh pool expects the FULL {GALAXY_MESH_SHAPE} mesh as its parent, got "
        f"{tuple(mesh_device.shape)}; opening a partial shape top-level is the fabric-init trap "
        f"(BRINGUP_RECIPE.md:1672-1693)"
    )
    pool = SubmeshPool(mesh_device)
    yield pool
    mesh_device.quiesce_devices()
