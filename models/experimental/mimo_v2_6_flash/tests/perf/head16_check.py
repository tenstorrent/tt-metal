# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GreedyHead and DeviceSampler with 16 decode users on Galaxy row 0, against the host: random fp32 logits
[32, vocab] split over the row's chips; every user's greedy token must be torch.argmax of its row (GreedyHead on bf16
and fp32 logits, DeviceSampler with greedy rows), and a sampled step must pass DeviceSampler.check. Prints
HEAD16_PASS."""

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.device_sampler import DeviceSampler
from models.experimental.mimo_v2_6_flash.tt.greedy_head import GreedyHead

VOCAB_LOCAL = 19072
USERS = 16


def main():
    torch.manual_seed(0)
    mesh, row = common.open_row()
    ok = True
    try:
        plan = MeshPlan(row)
        logits = torch.randn(32, VOCAB_LOCAL * plan.tp) * 3
        want = logits[:USERS].argmax(-1)
        for dtype in (ttnn.bfloat16, ttnn.float32):
            t = ttnn.from_torch(
                logits.reshape(1, 1, 32, -1),
                device=row,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=plan.shard(-1),
            )
            ref = logits.to(torch.bfloat16).float()[:USERS].argmax(-1) if dtype == ttnn.bfloat16 else want
            head = GreedyHead(plan, VOCAB_LOCAL, USERS, 32)
            got = ttnn.to_torch(ttnn.get_device_tensors(head(t))[0]).reshape(-1)[:USERS].to(torch.int64)
            bad = [u for u in range(USERS) if int(got[u]) != int(ref[u])]
            print(f"GreedyHead {dtype}: mismatched users {bad}", flush=True)
            ok &= not bad
            if dtype == ttnn.float32:
                sampler = DeviceSampler(plan, VOCAB_LOCAL, USERS, 32)
                got = ttnn.to_torch(ttnn.get_device_tensors(sampler(t))[0]).reshape(-1)[:USERS].to(torch.int64)
                bad = [u for u in range(USERS) if int(got[u]) != int(want[u])]
                print(f"DeviceSampler greedy: mismatched users {bad}", flush=True)
                ok &= not bad
                temp = torch.ones(USERS)
                top_k = torch.zeros(USERS, dtype=torch.int32)
                top_p = torch.full((USERS,), 0.9)
                hosts, _ = sampler.host_params(temp, top_k, top_p)
                sampler.set_params(hosts)
                rng = sampler.read_rng()
                tokens = ttnn.to_torch(ttnn.get_device_tensors(sampler(t))[0]).reshape(-1)
                checked, exact, mism = sampler.check(logits[:USERS], tokens, rng, [True] * USERS)
                print(f"DeviceSampler sampled: checked {checked}, exact {exact}, mismatches {mism}", flush=True)
                ok &= checked == USERS and not mism
            t.deallocate(True)
    finally:
        common.close(mesh, row)
    print("HEAD16_PASS" if ok else "HEAD16_FAIL", flush=True)


if __name__ == "__main__":
    main()
