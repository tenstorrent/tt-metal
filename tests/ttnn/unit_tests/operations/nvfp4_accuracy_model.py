# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only exact normal-domain NVFP4→BF16 SFPU accuracy model.

Run: python tests/ttnn/unit_tests/operations/nvfp4_accuracy_model.py --output report.json
No device execution. All model values are nonnegative; sign symmetry is
verified separately. Zero is exact and excluded from ULP normalization.
"""
import argparse
import json
from pathlib import Path
import numpy as np

MAGNITUDES = np.array([0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
SCALE_CODES = np.arange(1, 127, dtype=np.int32)
EXPONENTS = SCALE_CODES >> 3
MANTISSAS = SCALE_CODES & 7
SCALES = np.where(EXPONENTS == 0, MANTISSAS * 2.0**-9, (1 + MANTISSAS / 8) * 2.0 ** (EXPONENTS - 7))
PRODUCTS = (SCALES[:, None] * MAGNITUDES[None, :]).reshape(-1)
PRODUCT_SCALE_CODES = np.repeat(SCALE_CODES, len(MAGNITUDES))
PRODUCT_DATA = np.tile(MAGNITUDES, len(SCALES))


# All g lie in [1,2), so this is exact BF16 RNE; endpoints may round to 2.
def global_rne_bf16(g):
    return np.rint(g * 128) / 128


# Inputs here are normal FP64 values in the BF16 range. Using an FP64
# quantum avoids an intermediate FP32 rounding before the BF16 rounding.
def quantum(x):
    return np.exp2(np.floor(np.log2(np.abs(x))) - 7)


def rne_bf16(x):
    q = quantum(x)
    return np.copysign(np.rint(np.abs(x) / q) * q, x)


def rna_bf16(x):
    q = quantum(x)
    return np.copysign(np.floor(np.abs(x) / q + 0.5) * q, x)


# Positive BF16 code distance; sign symmetry is checked independently.
def positive_code(x):
    return x.astype(np.float32).view(np.uint32).astype(np.int64) >> 16


def evaluate_products(g, mode):
    g = np.asarray(g, dtype=np.float32).astype(np.float64)
    gb = global_rne_bf16(g)
    x = PRODUCTS[:, None] * g[None, :]
    y = rna_bf16(PRODUCTS[:, None] * gb[None, :])
    error = np.abs(y - x) / quantum(x)
    index = np.unravel_index(np.argmax(error), error.shape)
    i, j = index
    code_dist = np.abs(positive_code(y) - positive_code(rne_bf16(x)))
    negative_y = rna_bf16(-PRODUCTS[:, None] * gb[None, :])
    np.testing.assert_array_equal(negative_y, -y)
    return {
        "mode": mode,
        "product_encodings": int(len(PRODUCTS)),
        "global_scales": int(len(g)),
        "evaluated_cases": int(x.size),
        "max_bf16_ulp_error": float(error[index]),
        "mean_bf16_ulp_error": float(error.mean()),
        "median_bf16_ulp_error": float(np.median(error)),
        "p99_bf16_ulp_error": float(np.quantile(error, 0.99)),
        "max_bf16_code_distance_from_exact_rne": int(code_dist.max()),
        "sign_symmetry_verified": True,
        "worst": {
            "scale_code": int(PRODUCT_SCALE_CODES[i]),
            "scale": float(SCALES[PRODUCT_SCALE_CODES[i] - 1]),
            "datum": float(PRODUCT_DATA[i]),
            "product": float(PRODUCTS[i]),
            "global_scale": float(g[j]),
            "bf16_global_scale": float(gb[j]),
            "exact": float(x[index]),
            "model": float(y[index]),
            "bf16_quantum": float(quantum(x[index])),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output_path = parser.parse_args().output
    exact_globals = np.float32(1 + np.arange(128) / 128)
    midpoints = np.float32(1 + (np.arange(128) + 0.5) / 128)
    sweep512 = np.unique(
        np.concatenate(
            [
                exact_globals,
                midpoints,
                np.nextafter(midpoints, np.float32(-np.inf)),
                np.nextafter(midpoints, np.float32(np.inf)),
            ]
        )
    )
    assert len(sweep512) == 512

    # Extend each product's set independently. For fixed p, the model y is
    # constant between global BF16-rounding thresholds. Its error in BF16 ULPs
    # is piecewise linear in g, with extra discontinuities at x's binade
    # boundaries. Thus checking thresholds and adjacent FP32 values covers
    # maxima over all FP32 g∈[1,2) in this normal-domain model. Output rounding
    # midpoints are also included to cover changes in the exact-RNE reference.
    extended_base = np.unique(np.concatenate([sweep512, [np.nextafter(np.float32(2), np.float32(0))]]))
    extended_errors = []
    extended_count = 0
    max_code = 0
    worst_error = -1.0
    worst = None
    for i, p in enumerate(PRODUCTS):
        exponent = int(np.floor(np.log2(p)))
        thresholds = np.concatenate([(np.arange(128, 256) + 0.5) * 2.0 ** (k - 7) for k in [exponent, exponent + 1]])
        thresholds = np.concatenate([thresholds, [2.0 ** (exponent + 1)]])
        gs = (thresholds / p).astype(np.float32)
        gs = gs[(gs >= 1) & (gs < 2)]
        gs = np.unique(
            np.concatenate(
                [
                    extended_base,
                    gs,
                    np.nextafter(gs, np.float32(-np.inf)),
                    np.nextafter(gs, np.float32(np.inf)),
                ]
            )
        ).astype(np.float64)
        gs = gs[(gs >= 1) & (gs < 2)]
        gb = global_rne_bf16(gs)
        x = p * gs
        y = rna_bf16(p * gb)
        error = np.abs(y - x) / quantum(x)
        j = np.argmax(error)
        if error[j] > worst_error:
            worst_error = float(error[j])
            worst = {
                "scale_code": int(PRODUCT_SCALE_CODES[i]),
                "scale": float(SCALES[PRODUCT_SCALE_CODES[i] - 1]),
                "datum": float(PRODUCT_DATA[i]),
                "product": float(p),
                "global_scale": float(gs[j]),
                "bf16_global_scale": float(gb[j]),
                "exact": float(x[j]),
                "model": float(y[j]),
                "bf16_quantum": float(quantum(x[j])),
            }
        code_distance = np.abs(positive_code(y) - positive_code(rne_bf16(x)))
        max_code = max(max_code, int(code_distance.max()))
        np.testing.assert_array_equal(rna_bf16(-p * gb), -y)
        extended_errors.append(error)
        extended_count += len(gs)
    extended_errors = np.concatenate(extended_errors)
    report = {
        "provenance": "CPU-only NumPy model; no device or simulator execution",
        "formula": "RNA_BF16(d*s*RNE_BF16(g)) vs exact FP64 d*s*g32",
        "domain": "127 finite nonnegative scales; 7 nonzero positive data magnitudes; sign symmetry checked; g32 in [1,2); no underflow/overflow",
        "zero_cases": "Scale zero or datum zero produce zero; excluded from normalized ULP statistics",
        "numpy_version": np.__version__,
        "exact_bf16_globals_128": evaluate_products(exact_globals, "exact_bf16_globals"),
        "global_boundaries_512": evaluate_products(sweep512, "global_bf16_boundary_sweep"),
        "extended_thresholds": {
            "evaluated_cases": int(extended_count),
            "max_bf16_ulp_error": worst_error,
            "mean_bf16_ulp_error": float(extended_errors.mean()),
            "median_bf16_ulp_error": float(np.median(extended_errors)),
            "p99_bf16_ulp_error": float(np.quantile(extended_errors, 0.99)),
            "max_bf16_code_distance_from_exact_rne": max_code,
            "sign_symmetry_verified": True,
            "worst": worst,
            "statistics_weighting": "Each generated critical point counted once per encoded product; not a probability distribution over FP32 globals",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
