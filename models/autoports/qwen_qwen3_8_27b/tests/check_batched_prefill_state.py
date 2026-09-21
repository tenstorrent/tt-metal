# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device isolation/continuation check on real TP4 linear and full-attention layers."""

import json

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric


def snapshot(cache):
    return {
        (i, name, chip): ttnn.to_torch(part).clone()
        for i, state in enumerate(cache.layers)
        for name in ("conv", "recurrent", "key", "value")
        if (tensor := getattr(state, name)) is not None
        for chip, part in enumerate(ttnn.get_device_tensors(tensor))
    }


def main():
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen = None
    try:
        gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=[0, 3])
        cache = gen._ensure_cache(4, 128)
        prefix = torch.arange(32).repeat(4, 1) + torch.arange(4)[:, None] * 17 + 100
        results = []
        for grouped, perturb in [(False, False), (True, False), (True, True)]:
            gen._release_traces()
            gen.prefill_signatures.clear()
            gen.reset()
            gen.batched_prefill = False
            output = gen.prefill_forward(prefix, page_table=gen.page_table, kv_cache=cache, prompt_lens=[32] * 4)
            del output
            before = snapshot(cache)
            gen.batched_prefill = grouped
            tokens = torch.arange(33).repeat(2, 1) + torch.tensor([[200], [300]])
            if perturb:
                tokens[0] += 19
            output = gen.prefill_forward(
                tokens,
                page_table=gen.page_table,
                kv_cache=cache,
                prompt_lens=[33, 33],
                slots=[1, 2],
                start_pos=[32, 32],
            )
            logits = [gen._host_logits(x).clone() for x in output]
            del output
            after = snapshot(cache)
            for key, tensor in after.items():
                # Four pages per slot; untouched slots 0 and 3 must stay bitwise identical.
                inactive = [0, 3] if key[1] in ("conv", "recurrent") else list(range(4)) + list(range(12, 16))
                assert torch.equal(tensor[inactive], before[key][inactive]), ("inactive state changed", key)
            results.append((logits, after))
        reference, batched, perturbed = results
        worst_relative = 0.0
        for key, expected in reference[1].items():
            actual = batched[1][key]
            if key[1] in ("key", "value"):
                # Paged fill may overwrite causally masked rows after the logical end.
                def live(tensor):
                    rows = tensor.permute(0, 2, 1, 3).reshape(4, 128, -1)
                    return torch.cat([rows[i, : 65 if i in (1, 2) else 32] for i in range(4)])

                expected_live, actual_live = live(expected), live(actual)
            else:
                expected_live, actual_live = expected, actual
            relative = float(
                torch.linalg.vector_norm(actual_live.float() - expected_live.float())
                / torch.linalg.vector_norm(expected_live.float()).clamp_min(1e-10)
            )
            worst_relative = max(worst_relative, relative)
            assert relative < 0.01, (key, relative)
            if key[1] in ("conv", "recurrent"):
                peer_actual, peer_perturbed = actual[2], perturbed[1][key][2]
            else:
                peer_actual = actual.permute(0, 2, 1, 3).reshape(4, 128, -1)[2, :65]
                peer_perturbed = perturbed[1][key].permute(0, 2, 1, 3).reshape(4, 128, -1)[2, :65]
            assert torch.equal(peer_actual, peer_perturbed), ("cross-request state leak", key)
        correlations = []
        for expected, actual in zip(reference[0], batched[0]):
            correlation = float(
                torch.corrcoef(torch.stack([expected.float().flatten(), actual.float().flatten()]))[0, 1]
            )
            correlations.append(correlation)
            assert correlation > 0.999, correlation
            assert torch.equal(expected.argmax(-1), actual.argmax(-1))
        assert torch.equal(batched[0][1], perturbed[0][1]), "cross-request logits leak"
        print(
            "BATCHED_STATE_CHECK",
            json.dumps(
                dict(
                    passed=True,
                    logit_pcc=correlations,
                    worst_state_relative_l2=worst_relative,
                    inactive_slots_exact=True,
                    peer_isolation_exact=True,
                )
            ),
            flush=True,
        )
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
