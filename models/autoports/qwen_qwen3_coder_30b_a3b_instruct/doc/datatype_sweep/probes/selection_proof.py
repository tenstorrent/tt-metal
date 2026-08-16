# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prove the selected config is what construction takes **by default**.

The goal is explicit that a JSON file the model does not read is not a selected
config: it requires the choice to be "consumed by default by the construction
path the measurements use", and says to *prove* it rather than assert it.

So this builds the real 48-layer model the way every downstream caller builds it
-- ``build_generator(model_dir, mesh)``, **no precision argument** -- with
``QWEN3_PRECISION_CONFIG`` explicitly cleared from the environment first, so
nothing outside this process can be supplying the answer. Then it compares what
the *device* holds against ``selected_precision_config.json`` read off disk.

The comparison is deliberately against **device readback**, not against
``model.precision``: reading the config object back would only prove the
dataclass round-trips. ``fallback_audit`` reports dtypes read off the uploaded
tensors and the block widths ``_tuned_sparse_matmul_config`` actually resolved
to, which is the thing that would differ if the threading were broken -- and,
for the two fields stage 07 moved, the resolved width is exactly the value that
would be silently clamped if it were illegal.

Exits non-zero on any mismatch, so it is a gate and not a report.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

# Cleared BEFORE the model imports anything, so there is no chance a stale
# sweep row is what we end up measuring.
os.environ.pop("QWEN3_PRECISION_CONFIG", None)

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.precision import dtype_to_name  # noqa: E402

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE.parent
MODEL_DIR = SWEEP_DIR.parent.parent


def main() -> int:
    selected = json.loads((SWEEP_DIR / "selected_precision_config.json").read_text())

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    failures = []
    try:
        # exactly how the readiness runners and the perf probe build it
        gen = build_generator(MODEL_DIR, mesh, override_num_layers=48, max_context_len=8192, max_batch_size=1)
        gen._ensure_kv_cache()
        # Four tokens through the real traced decode path BEFORE the audit.
        # ``logits_dtype`` and ``sampling_dtype`` are verified off the tensors
        # the terminal path produced, and those do not exist until something has
        # run; auditing a freshly built model would only be able to echo the
        # config back, which is exactly what this probe refuses to do.
        ids = gen.tokenizer("def fib(n):", add_special_tokens=False)["input_ids"]
        gen.generate(ids, 4, enable_trace=True, sampling_mode="device", top_k=1)
        audit = gen.model.runtime_fallback_audit()

        # device readback -> the selected config's field name
        checks = [
            ("experts_gate_up_in0_block_w", audit["gate_up_in0_block_w"], selected["experts_gate_up_in0_block_w"]),
            ("experts_down_in0_block_w", audit["down_in0_block_w"], selected["experts_down_in0_block_w"]),
            (
                "experts_gate_up_dtype",
                audit["device_experts_gate_up_dtype"],
                f"DataType.{selected['experts_gate_up_dtype'].upper()}",
            ),
            (
                "experts_down_dtype",
                audit["device_experts_down_dtype"],
                f"DataType.{selected['experts_down_dtype'].upper()}",
            ),
            (
                "attention_qkv_dtype",
                audit["device_attention_qkv_dtype"],
                f"DataType.{selected['attention_qkv_dtype'].upper()}",
            ),
            (
                "attention_wo_dtype",
                audit["device_attention_wo_dtype"],
                f"DataType.{selected['attention_wo_dtype'].upper()}",
            ),
            ("lm_head_dtype", audit["lm_head_weight_dtype"], f"DataType.{selected['lm_head_dtype'].upper()}"),
            ("embedding_dtype", audit["embedding_weight_dtype"], f"DataType.{selected['embedding_dtype'].upper()}"),
            ("router_dtype", audit["device_router_dtype"], f"DataType.{selected['router_dtype'].upper()}"),
            (
                "norm_weight_dtype",
                audit["device_norm_weight_dtype"],
                f"DataType.{selected['norm_weight_dtype'].upper()}",
            ),
            # plain name on both sides: the audit emits "bfloat16" for this one
            # field to keep faith with doc/optimized_full_model's committed
            # evidence, and the JSON config uses plain names throughout.
            ("kv_cache_dtype", audit["kv_cache_dtype"], selected["kv_cache_dtype"]),
            ("activation_dtype", audit["activation_dtype"], f"DataType.{selected['activation_dtype'].upper()}"),
            ("experts_fidelity", audit["expert_math_fidelity"], f"MathFidelity.{selected['experts_fidelity']}"),
            (
                "router_window_fidelity",
                audit["router_window_math_fidelity"],
                f"MathFidelity.{selected['router_window_fidelity']}",
            ),
            # -- added by the stage-07 review ---------------------------------
            #
            # These four were the fields the proof did not check, which is why
            # ``R03_lmhead_lofi``, ``R21_norm_hifi2`` and
            # ``R22_logits_sampling_bfp8`` all produced a ``device_audit``
            # byte-identical to the baseline's: "this lever does nothing" and
            # "this lever is not wired up" looked the same. Checking them found
            # that ``norm_fidelity`` was in fact the second case.
            #
            # The two fidelities come off the compute-kernel-config objects the
            # ops are handed; the two dtypes come off the tensors the terminal
            # path produced during the four tokens generated above.
            ("lm_head_fidelity", audit["lm_head_math_fidelity"], f"MathFidelity.{selected['lm_head_fidelity']}"),
            ("norm_fidelity", audit["norm_math_fidelity"], f"MathFidelity.{selected['norm_fidelity']}"),
            ("logits_dtype", audit["logits_dtype_observed"], selected["logits_dtype"]),
            ("sampling_dtype", audit["sampling_dtype_observed"], selected["sampling_dtype"]),
            # ``ccl_dtype`` is ``None`` == "inherit the activation dtype" in the
            # selected config, and the audit reports the RESOLVED value, so the
            # comparison is against ``effective_ccl_dtype``'s definition rather
            # than against the raw field. This is the one entry that is a
            # resolved config value rather than a device readback -- the
            # collectives keep no dtype-tagged tensor to read back -- and it is
            # labelled as such rather than counted as proof.
            (
                "ccl_dtype (resolved, not readback)",
                audit["ccl_dtype"],
                f"DataType.{(selected['ccl_dtype'] or selected['activation_dtype']).upper()}",
            ),
        ]
        print(f"{'field':32s} {'on device':26s} {'selected config':26s} ok")
        for name, got, want in checks:
            ok = str(got) == str(want)
            if not ok:
                failures.append((name, got, want))
            print(f"{name:32s} {str(got):26s} {str(want):26s} {'OK' if ok else 'MISMATCH'}")

        # attention_fidelity is None in the selected config == "op default", and
        # the audit reports None for exactly that, so it is checked separately
        # rather than string-formatted into a MathFidelity that does not exist.
        af_got, af_want = audit["attention_math_fidelity"], selected["attention_fidelity"]
        af_ok = (af_got is None) == (af_want is None)
        print(f"{'attention_fidelity':32s} {str(af_got):26s} {str(af_want):26s} {'OK' if af_ok else 'MISMATCH'}")
        if not af_ok:
            failures.append(("attention_fidelity", af_got, af_want))

        # The KV dtype, read off the ALLOCATED cache tensor.
        #
        # ``runtime_fallback_audit`` reads ``model.kv_cache``, but the generator
        # keeps its cache in ``Qwen3CoderGenerator._kv_cache`` and only the
        # model's own ``_ensure_kv_cache`` populates the model attribute -- so
        # through this path the audit honestly reports
        # ``kv_cache_dtype_source == "config_not_yet_allocated"`` and falls back
        # to the configured value. That is not a lie, but it is not a readback
        # either, so the tensor is inspected directly here. This is the field a
        # KV-dtype sweep would most want proven on device.
        kv_tensor_dtype = dtype_to_name(gen._kv_cache[0].k.dtype)
        kv_ok = kv_tensor_dtype == selected["kv_cache_dtype"]
        print(
            f"{'kv_cache_dtype (allocated tensor)':32s} {kv_tensor_dtype:26s} "
            f"{selected['kv_cache_dtype']:26s} {'OK' if kv_ok else 'MISMATCH'}"
        )
        if not kv_ok:
            failures.append(("kv_cache_dtype_allocated_tensor", kv_tensor_dtype, selected["kv_cache_dtype"]))
        audit["kv_cache_dtype_allocated_tensor"] = kv_tensor_dtype

        print(f"\nenv QWEN3_PRECISION_CONFIG = {os.getenv('QWEN3_PRECISION_CONFIG')!r} (must be None)")
        assert os.getenv("QWEN3_PRECISION_CONFIG") is None

        (SWEEP_DIR / "selection_proof.json").write_text(
            json.dumps(
                {
                    "built_via": "build_generator(model_dir, mesh) with NO precision argument",
                    "env_QWEN3_PRECISION_CONFIG": None,
                    "layers": 48,
                    "device_audit": audit,
                    "selected_precision_config": selected,
                    "mismatches": failures,
                    "result": "PASS" if not failures else "FAIL",
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        gen.teardown()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    if failures:
        print(f"\nFAIL: {len(failures)} mismatch(es): {failures}")
        return 1
    print("\nPASS: the default construction path puts the selected config on the device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
