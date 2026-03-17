# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Dump tensors to formatted text for inspection and comparison.
"""

import os
import re
import numpy as np
import torch
import ttnn


def _tensor_to_numpy(tensor):
    """Convert torch or ttnn tensor to float32 numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().float().cpu().numpy()
    return ttnn.to_torch(tensor, dtype=torch.float32).detach().cpu().numpy()


def _format_tensor_rows_columns(data, fmt, max_elems, tail_note):
    """Format array body to match rows and columns, with column alignment."""
    if data.ndim == 1:
        parts = [fmt.format(float(x)) for x in data]
        return " ".join(parts) + tail_note

    if data.ndim == 2:
        rows = []
        elems_so_far = 0
        for i in range(data.shape[0]):
            if max_elems is not None and elems_so_far >= max_elems:
                rows.append("# ... (remaining rows omitted)")
                break
            row_vals = [float(x) for x in data[i]]
            elems_so_far += len(row_vals)
            rows.append([fmt.format(x) for x in row_vals])
        # Column alignment: use only list rows for alignment
        data_rows = [r for r in rows if isinstance(r, list)]
        if not data_rows:
            return "\n".join(str(r) for r in rows) + tail_note
        cols = len(data_rows[0])
        widths = [max(len(data_rows[j][c]) for j in range(len(data_rows))) for c in range(cols)]
        line_list = [" ".join(data_rows[j][c].rjust(widths[c]) for c in range(cols)) for j in range(len(data_rows))]
        if len(rows) > len(data_rows):
            line_list.append(rows[-1])  # append "... (remaining rows omitted)"
        return "\n".join(line_list) + tail_note

    # 3D+: print each 2D slice (e.g. data[k,:,:]) with blank line between
    out_parts = []
    elems_so_far = 0
    for k in range(data.shape[0]):
        if max_elems is not None and elems_so_far >= max_elems:
            out_parts.append("# ... (remaining slices omitted)")
            break
        slice_2d = data[k]
        slice_flat = slice_2d.flatten()
        elems_so_far += slice_flat.size
        if slice_2d.ndim == 1:
            out_parts.append(" ".join(fmt.format(float(x)) for x in slice_2d))
        else:
            rows = [[fmt.format(float(x)) for x in slice_2d[i]] for i in range(slice_2d.shape[0])]
            cols = len(rows[0])
            widths = [max(len(rows[j][c]) for j in range(len(rows))) for c in range(cols)]
            out_parts.append(
                "\n".join(" ".join(rows[j][c].rjust(widths[c]) for c in range(cols)) for j in range(len(rows)))
            )
        if k < data.shape[0] - 1:
            out_parts.append("")  # blank line between slices
    return "\n".join(out_parts) + tail_note


def dump_tensor(
    tensor,
    name=None,
    file_path=None,
    precision=8,
    max_elems=None,
    layout="rows",
):
    """
    Dump a torch or ttnn tensor to a formatted string (and optionally to a file)
    for inspection and comparison with other tensors.

    Output format:
      - Header: name, shape, dtype, min, max, mean, std
      - Body: values formatted to match tensor rows and columns (default),
        or flat/inline per layout.

    Args:
        tensor: A torch.Tensor or ttnn tensor.
        name: Optional label for the tensor (included in header and in default filename).
        file_path: If set, write the dump to this path. If a directory, filename
            is derived from name (or "tensor_dump.txt").
        precision: Number of decimal places for numeric values (default 8).
        max_elems: If set, only print this many elements for large tensors;
            shows first max_elems with a "... (N total)" line so comparisons
            still show structure. Default None = print all.
        layout: "rows" (default) format to match rows/columns; "flat" (one value
            per line); "inline" (compact like numpy).

    Returns:
        The formatted dump string.
    """
    data = _tensor_to_numpy(tensor)
    flat = data.flatten()
    n = flat.size
    name_str = name or "tensor"

    # Summary stats
    min_val = float(flat.min())
    max_val = float(flat.max())
    mean_val = float(flat.mean())
    std_val = float(flat.std())

    fmt = f"{{:.{precision}g}}"
    header_lines = [
        f"# name: {name_str}",
        f"# shape: {data.shape}",
        f"# dtype: {data.dtype}",
        f"# size: {n}",
        f"# min: {fmt.format(min_val)}",
        f"# max: {fmt.format(max_val)}",
        f"# mean: {fmt.format(mean_val)}",
        f"# std: {fmt.format(std_val)}",
        "",
    ]
    header = "\n".join(header_lines)

    if max_elems is not None and n > max_elems:
        values_to_show = flat[:max_elems]
        tail_note = f"\n# ... ({n - max_elems} more elements, {n} total)"
    else:
        values_to_show = flat
        tail_note = ""

    if layout == "flat":
        value_lines = [fmt.format(float(x)) for x in values_to_show]
        body = "\n".join(value_lines) + tail_note
    elif layout == "rows":
        # Format to match tensor rows and columns; align columns for readability
        body = _format_tensor_rows_columns(data, fmt, max_elems, tail_note)
    else:
        # inline: compact, like numpy
        with np.printoptions(
            precision=precision,
            threshold=np.inf if max_elems is None else max_elems,
            linewidth=200,
            suppress=True,
        ):
            body = np.array2string(values_to_show if values_to_show is flat else values_to_show, separator=", ")
        if tail_note:
            body = body + tail_note

    out = header + body

    if file_path:
        if os.path.isdir(file_path) or not os.path.splitext(str(file_path))[1]:
            os.makedirs(file_path, exist_ok=True)
            safe_name = re.sub(r"[^\w\-]", "_", name_str).strip("_") or "tensor_dump"
            file_path = os.path.join(file_path, f"{safe_name}.txt")
        with open(file_path, "w") as f:
            f.write(out)

    return out
