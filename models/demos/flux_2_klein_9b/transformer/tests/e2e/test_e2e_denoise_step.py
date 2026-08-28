# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CALL 1 -- `run_denoise_step`: Gate 1, Gate 2 and Gate 3.

Gate 1  every routed graduated stub is still real ttnn, not a torch fallback:
        each live `_stubs/<name>.py` is byte-identical to its own frozen
        `.last_good_native` / `.last_good_sharded` snapshot AND to the sha256
        `e2e_plan.json` certified, `_flux2_ttnn.py` likewise, Source B's
        `_runtime_fallbacks.json` is empty, every routed object was built by its
        stub module, and no forbidden host call appears in the hot path. A
        SHARDED body counts as native and is never rewritten to replication.

Gate 2  every one of the 18 is INVOKED inside the real forward, the exact number
        of times `e2e_plan.json routing.table` says, with its output feeding
        downstream computation on the way to `proj_out`. There is no
        call-everything-once sweep anywhere in the package -- asserted, not
        asserted-about.

Gate 3  PCC of the TT velocity prediction against the HF golden, on the
        byte-identical input, printed on its own line on pass AND fail.

What Call 1 actually is
-----------------------
This checkpoint's whole output: one denoise forward mapping (packed latents,
text embeddings, timestep, position ids) to the velocity prediction
`[1, S_img, 128]` the flow-match sampler consumes. There is no `generate()` and
no other head -- `forward` IS the reference callable, so the golden is
`model(...)` itself.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.flux_2_klein_9b.transformer.tt import pipeline as tt_pipeline
from models.demos.flux_2_klein_9b.transformer.tt import reference as tt_reference
from models.demos.flux_2_klein_9b.transformer.tt import stubs as tt_stubs

# Everything of ours that runs on the hot path, plus the two host-only helpers
# it is built from. `tt/reference.py` is deliberately NOT here: computing the
# golden is what it is for, and `e2e_plan.json::forbidden_in_hot_path` allows HF
# exactly there.
HOT_PATH_SOURCES = (
    "models/demos/flux_2_klein_9b_transformer/tt/pipeline.py",
    "models/demos/flux_2_klein_9b_transformer/tt/inputs.py",
    "models/demos/flux_2_klein_9b_transformer/tt/stubs.py",
)

# The whole emitted package, for the "no call-everything-once sweep" half of
# Gate 2 -- a sweep helper hiding in a demo or a test would count just as much.
PACKAGE_SOURCES = tuple(
    str(path.relative_to(tt_stubs.REPO_ROOT))
    for path in sorted((tt_stubs.REPO_ROOT / "models/demos/flux_2_klein_9b_transformer").rglob("*.py"))
)


def test_gate_1_graduated_bodies_are_unmodified_native_ttnn():
    """Host-only: no device is opened, so this fails fast and cheap."""
    report = tt_stubs.gate_1_report()

    assert report["graduated"] == 18, f"expected the 18 certified stubs, got {report['graduated']}"
    assert report["ok"], "Gate 1 problems:\n  " + "\n  ".join(report["problems"])
    assert report["runtime_fallbacks"] == {}, f"components fell back to torch: {report['runtime_fallbacks']}"

    for name, entry in report["modules"].items():
        assert entry["snapshot"] in ("last_good_native", "last_good_sharded"), f"{name}: {entry['snapshot']}"
        assert entry["identical"], f"{name}: live body differs from its {entry['snapshot']} snapshot"
        assert entry["has_build"], f"{name}: no callable build(device, torch_module)"

    sharded = sorted(n for n, e in report["modules"].items() if e["snapshot"] == "last_good_sharded")
    print(f"[flux2] Gate 1: 18 graduated bodies unmodified; {len(sharded)} mesh-sharded: {sharded}", flush=True)


def test_gate_1_hot_path_has_no_forbidden_host_call():
    """Host-only static half of "the hot path is pure ttnn"."""
    findings = tt_stubs.forbidden_source_scan(HOT_PATH_SOURCES)
    assert not findings, "forbidden host calls on the hot path:\n  " + "\n  ".join(findings)

    sweeps = [f for f in tt_stubs.forbidden_source_scan(PACKAGE_SOURCES) if "coverage-sweep" in f]
    assert not sweeps, "a call-everything-once helper exists:\n  " + "\n  ".join(sweeps)
    print(f"[flux2] Gate 1: {len(HOT_PATH_SOURCES)} hot-path sources clean, no sweep helper in the package", flush=True)


@pytest.mark.timeout(5400)
def test_gate_1_every_routed_object_was_built_by_its_stub(flux2_pipeline):
    """Each routed object's defining module really lives under Source B's `_stubs/`."""
    pipe = flux2_pipeline
    provenance = tt_stubs.provenance()

    for name in tt_pipeline.ROUTED_STUBS:
        objects = pipe.stub_objects[name]
        assert objects, f"{name}: routed by the plan but no object was built"
        records = provenance.get(name, [])
        assert len(records) == len(objects), f"{name}: {len(objects)} objects but {len(records)} provenance records"
        for record in records:
            assert record["builder_module"].startswith(tt_stubs.STUBS_PACKAGE), record
            assert record["from_stubs"], (
                f"{name}: built object {record['type_module']}.{record['type_name']} does not resolve "
                f"under {tt_stubs.STUBS_PACKAGE}"
            )

    # The four explicitly-assembled blocks subclass the graduated block types, so
    # `pipeline.dual_blocks` / `single_blocks` are flat lists of same-typed
    # elements that stack discovery can walk.
    assert len(pipe.dual_blocks) == pipe.dual_layers, (pipe.dual_layers, len(pipe.dual_blocks))
    assert len(pipe.single_blocks) == pipe.single_layers, (pipe.single_layers, len(pipe.single_blocks))
    for block in pipe.dual_blocks:
        assert isinstance(block, tt_pipeline.TtFlux2TransformerBlock), type(block)
    for block in pipe.single_blocks:
        assert isinstance(block, tt_pipeline.TtFlux2SingleTransformerBlock), type(block)

    built = sum(len(v) for v in pipe.stub_objects.values())
    print(f"[flux2] Gate 1: {built} routed objects, all built by their graduated stub module", flush=True)


@pytest.mark.timeout(5400)
def test_gate_2_and_3_denoise_step(flux2_pipeline, flux2_inputs, flux2_reference, flux2_device, flux2_shapes):
    """CALL 1: run the chained forward, count the stubs, PCC against the golden."""
    pipe = flux2_pipeline
    depth = pipe.depth()
    print(
        f"[flux2] Call 1 at dual {depth['dual_layers']}/{pipe.full_dual_layers}, "
        f"single {depth['single_layers']}/{pipe.full_single_layers}, "
        f"S_txt={flux2_inputs['meta']['S_txt']} S_img={flux2_inputs['meta']['S_img']}",
        flush=True,
    )

    pipe.reset_invocations()
    sample = tt_pipeline.run_denoise_step(pipe, flux2_inputs)
    tt_sample = tt_pipeline.to_torch(sample, flux2_device).to(torch.float32)

    # ---- Gate 2: every graduated stub reached, exactly as often as routed -----
    expected = pipe.expected_calls_per_step()
    missing = sorted(name for name, count in pipe.invocations.items() if count == 0)
    assert not missing, f"graduated stubs never reached by the forward: {missing}"
    wrong = {
        name: (count, pipe.invocations[name]) for name, count in expected.items() if pipe.invocations[name] != count
    }
    assert not wrong, f"per-step call counts differ from routing.table (expected, actual): {wrong}"
    assert set(pipe.invocations) == set(tt_stubs.GRADUATED), (
        sorted(set(tt_stubs.GRADUATED) - set(pipe.invocations)),
        sorted(set(pipe.invocations) - set(tt_stubs.GRADUATED)),
    )
    print(
        f"[flux2] Gate 2: all 18 graduated stubs invoked, counts {dict(sorted(pipe.invocations.items()))}", flush=True
    )

    # ---- Gate 3: PCC against the HF golden on the identical input -------------
    golden = tt_reference._hf_reference_denoise_step(
        flux2_reference,
        flux2_inputs,
        dual_layers=depth["dual_layers"],
        single_layers=depth["single_layers"],
    )
    assert tuple(tt_sample.shape) == tuple(golden.shape), (tuple(tt_sample.shape), tuple(golden.shape))

    ok, achieved_pcc = comp_pcc(golden, tt_sample, flux2_shapes["pcc"])
    print(f"e2e PCC={achieved_pcc}", flush=True)
    assert ok, f"Call 1 e2e PCC {achieved_pcc} below the required {flux2_shapes['pcc']}"
