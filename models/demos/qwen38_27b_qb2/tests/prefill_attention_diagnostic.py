# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Controlled prefill numerical diagnostic; completion never qualifies a model."""

import math

CASES = ((3, 32, 65), (3, 32, 64), (3, 0, 65), (2, 0, 128))
VARIANTS = (
    "native",
    "accurate_exp",
    "fp32_accum",
    "accurate_exp_fp32",
    "hifi4_accurate_fp32",
    "native_repeat",
)


def configuration(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown prefill numerical diagnostic variant")
    accurate = variant in ("accurate_exp", "accurate_exp_fp32", "hifi4_accurate_fp32")
    fp32 = variant in ("fp32_accum", "accurate_exp_fp32", "hifi4_accurate_fp32")
    return dict(
        exp_approx_mode=False if accurate else None,
        compute_kernel=None
        if not fp32
        else dict(
            math_fidelity="HiFi4" if variant == "hifi4_accurate_fp32" else "HiFi2",
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )


def selected_reference(query, caches, table, start, selected):
    """FP64 selected-row causal attention plus an independent Torch SDPA check.

    Inputs are downloaded quantized KV and original BF16 Q. Neither raw
    unquantized KV nor a different causal convention is used as the reference.
    This diagnostic is deliberately restricted to small prefixes.
    """
    import torch
    import torch.nn.functional as F

    manual, independently_checked = [], []
    query_rows = query[:, :, selected, :].float()
    allowed = torch.arange(table.shape[-1] * 32)[None, :] <= start + torch.tensor(selected)[:, None]
    for user in range(query.shape[0]):
        key, value = [cache[table[user]].permute(1, 0, 2, 3).flatten(1, 2).float() for cache in caches]
        q = query_rows[user]
        scale = query.shape[-1] ** -0.5
        scores = q.double() @ key.double().transpose(-1, -2) * scale
        ref = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1) @ value.double()
        independent = F.scaled_dot_product_attention(q, key, value, attn_mask=allowed, scale=scale)
        if not torch.allclose(ref.float(), independent, atol=2e-6, rtol=2e-5):
            raise ValueError("Independent CPU reference disagrees with the selected-row causal calculation")
        manual.append(ref.float())
        independently_checked.append(independent)
    return torch.stack(manual), torch.stack(independently_checked)


def rank_hashes(values):
    return (
        isinstance(values, list)
        and len(values) == 4
        and all(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in values)
    )


def validate_report(report):
    if report.get("state") != "completed" or report.get("cleanup_completed") is not True:
        raise ValueError("Numerical diagnostic did not complete cleanly")
    if len(set(report.get("device_ids", []))) != 4:
        raise ValueError("Diagnostic requires four physical ranks")
    cases = report.get("cases", [])
    if [(r.get("batch"), r.get("start_pos"), r.get("chunk_tokens")) for r in cases] != list(CASES):
        raise ValueError("Numerical diagnostic geometry is incomplete")
    findings = []
    for row in cases:
        if row.get("independent_reference_checked") is not True:
            raise ValueError("Diagnostic reference is not independently checked")
        arms = row.get("arms", [])
        if [a.get("name") for a in arms] != list(VARIANTS):
            raise ValueError("Numerical diagnostic variants are incomplete")
        if arms[0].get("output_sha256_per_rank") != arms[-1].get("output_sha256_per_rank"):
            raise ValueError("Native before/after outputs changed")
        if arms[0].get("accuracy_per_rank") != arms[-1].get("accuracy_per_rank"):
            raise ValueError("Native before/after numerical evidence changed")
        passing, errors = [], {}
        for arm in arms:
            if arm.get("configuration") != configuration(arm["name"]):
                raise ValueError("Diagnostic configuration differs from the frozen plan")
            hashes = arm.get("output_sha256_per_rank")
            if not rank_hashes(hashes) or len(set(hashes)) != 1:
                raise ValueError("Diagnostic ranks disagree")
            if arm.get("cache_unchanged") is not True or arm.get("query_unchanged") is not True:
                raise ValueError("Diagnostic modified its fixed inputs")
            if len(row.get("expected_cache_sha256", [])) != 2 or arm.get("cache_sha256_per_rank") != [
                [h] * 4 for h in row["expected_cache_sha256"]
            ]:
                raise ValueError("Diagnostic cache hashes do not match the expected prefix and writes")
            checks = arm.get("accuracy_per_rank", [])
            if len(checks) != 4:
                raise ValueError("Missing per-rank numerical results")
            verdicts = []
            rms = []
            for check in checks:
                correlations, relative = check.get("pcc_per_user", []), check.get("relative_rms_per_user", [])
                if (
                    len(correlations) != row["batch"]
                    or len(relative) != row["batch"]
                    or any(not math.isfinite(v) for v in correlations + relative)
                    or any(v < 0 for v in relative)
                ):
                    raise ValueError("Incomplete or nonfinite per-user diagnostic values")
                verdict = all(v >= 0.999 for v in correlations) and all(v <= 0.02 for v in relative)
                if check.get("passed") != verdict:
                    raise ValueError("Diagnostic must retain the original numerical gate")
                verdicts.append(verdict)
                rms.extend(relative)
            errors[arm["name"]] = max(rms)
            if all(verdicts):
                passing.append(arm["name"])
        findings.append(
            dict(
                batch=row["batch"],
                start_pos=row["start_pos"],
                chunk_tokens=row["chunk_tokens"],
                passing_variants=passing,
                max_relative_rms=errors,
            )
        )
    return dict(diagnostic_completed=True, findings=findings, model_qualified=False, performance_gain_measured=False)
