# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC visualization utilities for DeepSeek prefill transformer tests.

Provides:
- Matplotlib PNG plots of per-layer PCC values
- Mermaid xychart-beta chart generation for GitHub Actions summaries
- Summary file writer for CI integration
"""

import os
from pathlib import Path

from loguru import logger


def _build_run_name(result: dict) -> str:
    """Derive a short run name from result metadata."""
    parts = []
    if "num_layers" in result:
        parts.append(f"{result['num_layers']}L")
    if "isl_total" in result:
        parts.append(f"isl{result['isl_total']}")
    if "weight_type" in result:
        parts.append(result["weight_type"])
    if "input_source" in result:
        parts.append(result["input_source"])
    if "mesh_shape" in result:
        ms = result["mesh_shape"]
        if isinstance(ms, (list, tuple)):
            parts.append(f"{ms[0]}x{ms[1]}")
        else:
            parts.append(str(ms))
    if "n_routed_experts" in result:
        parts.append(f"e{result['n_routed_experts']}")
    if "capacity_factor" in result:
        parts.append(f"cf{result['capacity_factor']}")
    return "_".join(parts) if parts else "run"


def generate_pcc_plots(result: dict, output_dir: str = "/tmp/pcc_plots") -> dict:
    """
    Generate Matplotlib PNG plots for PCC results.

    Args:
        result: Dict with 'pcc' tuple of (output_pcc, kvpe_kv_pcc, kvpe_pe_pcc) dicts,
                plus metadata keys (num_layers, isl_total, weight_type, etc.)
        output_dir: Directory to save PNG files.

    Returns:
        Stats dict with min/max/mean PCC values per category.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping PNG plot generation")
        return {}

    output_pcc, kvpe_kv_pcc, kvpe_pe_pcc = result["pcc"]
    run_name = _build_run_name(result)
    threshold = result.get("threshold", 0.99)
    kv_threshold = result.get("kv_threshold", threshold)
    pe_threshold = result.get("pe_threshold", threshold)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stats = {}

    # --- Subplot 1: Output PCC ---
    has_output = len(output_pcc) > 0
    has_kvpe = len(kvpe_kv_pcc) > 0 or len(kvpe_pe_pcc) > 0
    n_subplots = (1 if has_output else 0) + (1 if has_kvpe else 0)

    if n_subplots == 0:
        logger.warning("No PCC data to plot")
        return stats

    fig, axes = plt.subplots(1, n_subplots, figsize=(7 * n_subplots, 5))
    if n_subplots == 1:
        axes = [axes]

    ax_idx = 0

    if has_output:
        ax = axes[ax_idx]
        ax_idx += 1
        labels = list(output_pcc.keys())
        values = np.array(list(output_pcc.values()))
        x = np.arange(len(labels))

        colors = ["red" if v <= threshold else "steelblue" for v in values]
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(y=threshold, color="orange", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(l) for l in labels], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("PCC")
        ax.set_title(f"Output PCC — {run_name}")
        ax.legend(fontsize=8)
        ax.set_ylim(min(values.min() - 0.01, threshold - 0.02), 1.005)

        stats["output"] = {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())}

    if has_kvpe:
        ax = axes[ax_idx]
        kv_labels = list(kvpe_kv_pcc.keys())
        kv_values = np.array(list(kvpe_kv_pcc.values())) if kvpe_kv_pcc else np.array([])
        pe_labels = list(kvpe_pe_pcc.keys())
        pe_values = np.array(list(kvpe_pe_pcc.values())) if kvpe_pe_pcc else np.array([])

        n_kv = len(kv_labels)
        n_pe = len(pe_labels)
        n_total = max(n_kv, n_pe)
        x = np.arange(n_total)
        width = 0.35

        if n_kv > 0:
            kv_colors = ["red" if v <= kv_threshold else "steelblue" for v in kv_values]
            ax.bar(
                x[:n_kv] - width / 2, kv_values, width, color=kv_colors, edgecolor="black", linewidth=0.3, label="KV"
            )
        if n_pe > 0:
            pe_colors = ["red" if v <= pe_threshold else "seagreen" for v in pe_values]
            ax.bar(
                x[:n_pe] + width / 2, pe_values, width, color=pe_colors, edgecolor="black", linewidth=0.3, label="PE"
            )

        ax.axhline(y=kv_threshold, color="orange", linestyle="--", linewidth=1.2, label=f"KV thresh={kv_threshold}")
        if pe_threshold != kv_threshold:
            ax.axhline(y=pe_threshold, color="purple", linestyle="--", linewidth=1.2, label=f"PE thresh={pe_threshold}")
        combined_labels = kv_labels if n_kv >= n_pe else pe_labels
        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(l) for l in combined_labels], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("PCC")
        ax.set_title(f"KVPE Cache PCC — {run_name}")
        ax.legend(fontsize=8)

        all_vals = np.concatenate([v for v in [kv_values, pe_values] if len(v) > 0])
        min_thresh = min(kv_threshold, pe_threshold)
        ax.set_ylim(min(all_vals.min() - 0.01, min_thresh - 0.02), 1.005)

        if n_kv > 0:
            stats["kv"] = {
                "min": float(kv_values.min()),
                "max": float(kv_values.max()),
                "mean": float(kv_values.mean()),
            }
        if n_pe > 0:
            stats["pe"] = {
                "min": float(pe_values.min()),
                "max": float(pe_values.max()),
                "mean": float(pe_values.mean()),
            }

    plt.tight_layout()
    png_path = out_path / f"pcc_{run_name}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info(f"PCC plot saved to {png_path}")

    return stats


def _short_label(label: str) -> str:
    """Convert e.g. 'layer_0' -> '0', 'layer_5_kvpe_kv' -> '5_kv', 'norm' -> 'norm'."""
    label = label.replace("layer_", "").replace("_kvpe", "")
    return label


def generate_pcc_mermaid(result: dict, threshold: float = 0.99) -> str:
    """
    Generate Mermaid xychart-beta charts for PCC results.

    Produces two charts:
    1. Output PCC — single bar series (per-layer output + norm)
    2. KVPE Cache PCC — two bar series (KV and PE) on shared layer x-axis

    Args:
        result: Dict with 'pcc' tuple and metadata.
        threshold: PCC threshold for pass/fail.

    Returns:
        Markdown string with Mermaid chart blocks and summary table.
    """
    output_pcc, kvpe_kv_pcc, kvpe_pe_pcc = result["pcc"]
    run_name = _build_run_name(result)
    kv_threshold = result.get("kv_threshold", threshold)
    pe_threshold = result.get("pe_threshold", threshold)

    sections = []
    sections.append(f"## PCC Results — {run_name}\n")

    # Chart 1: Output PCC — red line + orange threshold
    if output_pcc:
        labels = list(output_pcc.keys())
        values = list(output_pcc.values())
        short_labels = [_short_label(l) for l in labels]

        sections.append("### Output PCC\n")
        sections.append("```mermaid")
        sections.append(
            "%%{init: {'theme': 'base', 'xyChart': {'width': 1600, 'height': 400}, 'themeVariables': {'xyChart': {'plotColorPalette': '#e74c3c, #e67e22'}}}}%%"
        )
        sections.append("xychart-beta")
        sections.append(f'  title "Output PCC — {run_name}"')
        sections.append(f"  x-axis [{', '.join(short_labels)}]")
        sections.append(f'  y-axis "PCC" {_y_range(values, threshold)}')
        sections.append(f"  line [{', '.join(f'{v:.4f}' for v in values)}]")
        sections.append(f"  line [{', '.join(str(threshold) for _ in values)}]")
        sections.append("```")
        sections.append(f"> 🔴 Output PCC &nbsp; 🟠 Threshold ({threshold})\n")

    # Chart 2: KVPE Cache PCC — green (KV) + blue (PE) lines + orange threshold
    if kvpe_kv_pcc or kvpe_pe_pcc:
        n_layers = max(len(kvpe_kv_pcc), len(kvpe_pe_pcc))
        layer_labels = [str(i) for i in range(n_layers)]
        kv_values = list(kvpe_kv_pcc.values())[:n_layers]
        pe_values = list(kvpe_pe_pcc.values())[:n_layers]

        all_vals = kv_values + pe_values

        # Build palette: green for KV, blue for PE, orange for KV threshold, purple for PE threshold
        palette_colors = []
        if kv_values:
            palette_colors.append("#2ecc71")
        if pe_values:
            palette_colors.append("#3498db")
        palette_colors.append("#e67e22")
        if kv_threshold != pe_threshold:
            palette_colors.append("#9b59b6")

        min_thresh = min(kv_threshold, pe_threshold)
        sections.append("### KVPE Cache PCC\n")
        sections.append("```mermaid")
        sections.append(
            "%%{init: {'theme': 'base', 'xyChart': {'width': 1600, 'height': 400}, 'themeVariables': {'xyChart': {'plotColorPalette': '"
            + ", ".join(palette_colors)
            + "'}}}}%%"
        )
        sections.append("xychart-beta")
        sections.append(f'  title "KVPE Cache PCC — {run_name}"')
        sections.append(f"  x-axis [{', '.join(layer_labels)}]")
        sections.append(f'  y-axis "PCC" {_y_range(all_vals, min_thresh)}')
        if kv_values:
            sections.append(f"  line [{', '.join(f'{v:.4f}' for v in kv_values)}]")
        if pe_values:
            sections.append(f"  line [{', '.join(f'{v:.4f}' for v in pe_values)}]")
        sections.append(f"  line [{', '.join(str(kv_threshold) for _ in range(n_layers))}]")
        if kv_threshold != pe_threshold:
            sections.append(f"  line [{', '.join(str(pe_threshold) for _ in range(n_layers))}]")
        sections.append("```")
        legend = f"> 🟢 KV Cache &nbsp; 🔵 PE Cache &nbsp; 🟠 KV Threshold ({kv_threshold})"
        if kv_threshold != pe_threshold:
            legend += f" &nbsp; 🟣 PE Threshold ({pe_threshold})"
        sections.append(legend + "\n")

    # Summary table — one row per layer with output, KV, and PE columns
    # Build lookup dicts keyed by layer index
    output_by_label = {_short_label(l): (pcc, threshold) for l, pcc in output_pcc.items()}
    kv_by_idx = {i: (pcc, kv_threshold) for i, pcc in enumerate(kvpe_kv_pcc.values())}
    pe_by_idx = {i: (pcc, pe_threshold) for i, pcc in enumerate(kvpe_pe_pcc.values())}

    # Collect all row labels: output labels + any extra KV/PE layers
    n_layers = max(len(kvpe_kv_pcc), len(kvpe_pe_pcc), 0)
    row_labels = list(output_by_label.keys())
    # Add layer indices not already covered by output labels
    for i in range(n_layers):
        if str(i) not in row_labels:
            row_labels.append(str(i))

    if row_labels:

        def _cell(pcc, thresh):
            status = "PASS" if pcc > thresh else ("FAIL" if pcc >= 0 else "ERROR")
            icon = "✅" if status == "PASS" else "❌"
            return f"{pcc:.4f}", str(thresh), f"{icon} {status}", pcc <= thresh

        sections.append("### Summary\n")
        sections.append(
            "| Stage | PCC | Threshold | Status | KV PCC | KV Threshold | KV Status | PE PCC | PE Threshold | PE Status |"
        )
        sections.append(
            "|-------|-----|-----------|--------|--------|--------------|-----------|--------|--------------|-----------|"
        )
        failures = 0
        all_values = []
        for label in row_labels:
            # Output column
            if label in output_by_label:
                o_pcc, o_thresh = output_by_label[label]
                o_val, o_th, o_st, o_fail = _cell(o_pcc, o_thresh)
                all_values.append(o_pcc)
                if o_fail:
                    failures += 1
            else:
                o_val, o_th, o_st = "", "", ""

            # KV column
            idx = int(label) if label.isdigit() else -1
            if idx in kv_by_idx:
                kv_pcc, kv_th_val = kv_by_idx[idx]
                kv_val, kv_th, kv_st, kv_fail = _cell(kv_pcc, kv_th_val)
                all_values.append(kv_pcc)
                if kv_fail:
                    failures += 1
            else:
                kv_val, kv_th, kv_st = "", "", ""

            # PE column
            if idx in pe_by_idx:
                pe_pcc, pe_th_val = pe_by_idx[idx]
                pe_val, pe_th, pe_st, pe_fail = _cell(pe_pcc, pe_th_val)
                all_values.append(pe_pcc)
                if pe_fail:
                    failures += 1
            else:
                pe_val, pe_th, pe_st = "", "", ""

            sections.append(
                f"| {label} | {o_val} | {o_th} | {o_st} | {kv_val} | {kv_th} | {kv_st} | {pe_val} | {pe_th} | {pe_st} |"
            )

        sections.append(
            f"\n**Min PCC**: {min(all_values):.6f} | **Mean PCC**: {sum(all_values)/len(all_values):.6f} | **Failures**: {failures}/{len(all_values)}"
        )

    return "\n".join(sections)


def _y_range(values: list, threshold: float) -> str:
    """Compute y-axis range string for mermaid chart."""
    min_val = min(min(values), threshold) - 0.02
    return f"{max(0, min_val):.2f} --> 1.00"


def write_pcc_summary(result: dict, threshold: float = 0.99, output_dir: str = None) -> Path:
    """
    Write PCC summary markdown (with Mermaid charts) to a per-run file.

    Each run gets its own file named ``pcc_summary_{run_name}.md`` so that
    multiple parameterized runs in CI don't overwrite each other.  The CI
    publish step globs all files in the directory and concatenates them into
    ``$GITHUB_STEP_SUMMARY``.

    Args:
        result: Dict with 'pcc' tuple and metadata.
        threshold: PCC threshold for pass/fail.
        output_dir: Directory for summary files.
                    Defaults to PCC_SUMMARY_DIR env var or /tmp/pcc_summaries.

    Returns:
        Path to the written file.
    """
    if output_dir is None:
        output_dir = os.getenv("PCC_SUMMARY_DIR", "/tmp/pcc_summaries")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_name = _build_run_name(result)
    path = out / f"pcc_summary_{run_name}.md"

    content = generate_pcc_mermaid(result, threshold=threshold)
    path.write_text(content)
    logger.info(f"PCC summary written to {path}")

    return path
