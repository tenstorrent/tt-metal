# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch


def bitonic_sort(a, indices, up=True):
    def comp_and_swap(i, j, dir):
        if dir == (a[i] > a[j]):
            a[i], a[j] = a[j], a[i]
            indices[i], indices[j] = indices[j], indices[i]

    def bitonic_merge(low, cnt, dir):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                comp_and_swap(i, i + k, dir)
            bitonic_merge(low, k, dir)
            bitonic_merge(low + k, k, dir)

    def bitonic_sort_rec(low, cnt, dir):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort_rec(low, k, True)
            bitonic_sort_rec(low + k, k, False)
            bitonic_merge(low, cnt, dir)

    bitonic_sort_rec(0, len(a), up)


def topk_bitonic(input_tensor, k, dim=-1, largest=True, sorted=True):
    assert dim == -1, "This custom bitonic topK only supports last dimension"
    orig_shape = input_tensor.shape
    last_dim = orig_shape[dim]

    if (last_dim & (last_dim - 1)) != 0:
        raise ValueError("The last dimension must be a power of 2.")

    flat = input_tensor.reshape(-1, last_dim)
    topk_vals = []
    topk_indices = []

    for row_idx, row in enumerate(flat):
        row_vals = row.tolist()
        row_indices = list(range(len(row_vals)))

        bitonic_sort(row_vals, row_indices, up=not largest)

        selected = list(zip(row_vals[:k], row_indices[:k]))
        if sorted:
            selected.sort(reverse=largest, key=lambda x: x[0])
        vals_row, inds_row = zip(*selected)

        topk_vals.append(torch.tensor(vals_row, dtype=input_tensor.dtype))
        topk_indices.append(torch.tensor(inds_row, dtype=torch.long))

    # Stack and reshape
    topk_vals = torch.stack(topk_vals).reshape(*orig_shape[:-1], k)
    topk_indices = torch.stack(topk_indices).reshape(*orig_shape[:-1], k)
    return topk_vals, topk_indices
