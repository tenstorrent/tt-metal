# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Protocol test of the product adapter (generator_vllm.py) against a FAKE engine: no device. Run in the image (needs vllm_tt_plugin + the device-free ttnn import):
prefill_forward -> propose_draft_tokens -> verify decode_forward -> narrow decode_forward -> release. Checks shapes / dtypes / routing the plugin contract demands (docs/SPEC_DECODE_CONTRACT.md)."""
import os
import types

os.environ.setdefault("HY4_SPEC_K", "3")
import torch

from models.demos.blackhole.hy4_preview.tt import generator_vllm as g

K, R = 3, 4


class FakeEng:
    spec = True
    sampled = True
    vocab = 120832
    pages = 129
    dl = None

    def __init__(self):
        self.K = K
        self.live = [False] * R
        self.pend = [0] * R
        self.pos = [0] * R
        self.drafts = [[-1] * K for _ in range(R)]
        self.temp = [0.0] * R
        self.seeds = [0] * R
        self.rounds, self.t_round = 0, 0.0
        self.log = []

    def prefill(self, prompts, sampling):
        out = {}
        for r, p in prompts.items():
            out[r] = 100 + r
            self.live[r], self.pend[r], self.pos[r], self.drafts[r] = True, 100 + r, len(p), [1, 2, 3]
        self.log.append(("prefill", sorted(prompts), dict(sampling)))
        return out

    def round(self, caps, want, temps, seeds, plugin_block=None):
        self.log.append(("round", list(caps), want, sorted(plugin_block or {})))
        st = []
        for r in range(R):
            m = min(caps[r], 2) if self.live[r] else 0
            a = [10 * (r + 1) + j for j in range(K + 1)]
            if self.live[r]:
                self.pend[r], self.pos[r] = a[m], self.pos[r] + m + 1
                self.drafts[r] = [4, 5, 6]
            st.append(dict(a=a, m=m, pos=self.pos[r], drafts=[4, 5, 6], tok=[a[m], 4, 5, 6]))
        self.rounds += 1
        return st

    def release(self, r):
        self.live[r] = False


def main():
    m = g.HYV4ForCausalLM.__new__(g.HYV4ForCausalLM)
    m.eng = FakeEng()
    m.slots = g._Slots(R)
    m.max_num_seqs, m.max_seq_len, m.vocab, m.pages = R, 65536, 120832, 129
    m.steps, m._seeds, m._samp_set, m._warned_topk, m._active = 0, [0] * R, False, False, [0] * R
    sp = types.SimpleNamespace(
        temperature=[0.0, 0.8, 0.0, 0.0],
        top_k=[-1] * 4,
        top_p=[1.0] * 4,
        seed=[None, 5, None, None],
        enable_log_probs=[False] * 4,
    )
    toks = torch.randint(1, 1000, (2, 10), dtype=torch.int32)
    first = m.prefill_forward(toks, prompt_lens=[10, 7], empty_slots=[0, 1], sampling_params=sp)
    assert first.dtype == torch.int32 and first.shape == (2,) and first.tolist() == [100, 101], first
    # propose after the prefill commit: committed block [B, 1+K], count 1, position of the first token = prompt length
    ct = torch.full((4, 4), -1, dtype=torch.int32)
    ct[0, 0], ct[1, 0] = 100, 101
    cp = torch.full((4, 4), -1, dtype=torch.int32)
    cp[0] = torch.tensor([10, 11, 12, 13])
    cp[1] = torch.tensor([7, 8, 9, 10])
    d = m.propose_draft_tokens(3, ct, cp, torch.ones(4, dtype=torch.int32))
    assert d.draft_token_ids.shape == (4, 3) and d.draft_token_ids.dtype == torch.int32
    assert d.num_valid.tolist() == [3, 3, 0, 0], d.num_valid
    # verify block
    blk = torch.zeros(4, 4, dtype=torch.int32)
    blk[0] = torch.tensor([100, 1, 2, 3])
    blk[1] = torch.tensor([101, 1, 2, 3])
    pos = torch.full((4, 4), -1, dtype=torch.int32)
    pos[0] = torch.tensor([10, 11, 12, 13])
    pos[1] = torch.tensor([7, 8, 9, 10])
    nvd = torch.tensor([3, 3, 0, 0], dtype=torch.int32)
    out = m.decode_forward(
        blk,
        pos,
        num_valid_drafts=nvd,
        accepted_counts=torch.ones(4, dtype=torch.int32),
        spec_mode="argmax_ids",
        sampling_params=sp,
    )
    assert out.spec_mode == "argmax_ids" and out.argmax_ids.shape == (4, 4) and out.argmax_ids.dtype == torch.int32
    assert out.argmax_ids[0].tolist() == [10, 11, 12, 13] and out.argmax_ids[2].tolist() == [0, 0, 0, 0]
    assert m.eng.log[-1] == ("round", [2 + 1, 3, 0, 0], True, [0, 1]) or m.eng.log[-1][1] == [3, 3, 0, 0], m.eng.log[-1]
    # propose after the verify: committed block = accepted drafts + correction; row 0 accepted m=2 -> last = a_2 = 12 at position 10 + 2 + 1 = 13
    ct2 = torch.full((4, 4), -1, dtype=torch.int32)
    ct2[0, :3] = torch.tensor([1, 2, 12])
    ct2[1, :3] = torch.tensor([1, 2, 22])
    cp2 = torch.full((4, 4), -1, dtype=torch.int32)
    cp2[0] = torch.tensor([11, 12, 13, 14])
    cp2[1] = torch.tensor([8, 9, 10, 11])
    d2 = m.propose_draft_tokens(3, ct2, cp2, torch.tensor([3, 3, 1, 1], dtype=torch.int32))
    assert d2.num_valid.tolist() == [3, 3, 0, 0], (d2.num_valid, m.eng.pend, m.eng.pos)
    # narrow decode: [B,1] tokens, 1-D positions, sampling params present
    nt = torch.tensor([[12], [22], [0], [0]], dtype=torch.int32)
    np_ = torch.tensor([13, 10, -1, -1], dtype=torch.int32)
    o = m.decode_forward(nt, np_, sampling_params=sp)
    assert o.shape == (4, 1) and o.dtype == torch.int32, (o.shape, o.dtype)
    assert m.eng.log[-1][1] == [0, 0, 0, 0], m.eng.log[-1]
    # release
    m.release_request(0)
    assert not m.eng.live[0]
    # live cap switch file
    open("/tmp/spec_cap_test", "w").write("0")
    os.environ["HY4_SPEC_CAP_FILE"] = "/tmp/spec_cap_test"
    d3 = m.propose_draft_tokens(3, ct2, cp2, torch.tensor([3, 3, 1, 1], dtype=torch.int32))
    assert d3.num_valid.tolist() == [0, 0, 0, 0], d3.num_valid
    print("ADAPTER_PROTOCOL_TEST_OK")


if __name__ == "__main__":
    main()
