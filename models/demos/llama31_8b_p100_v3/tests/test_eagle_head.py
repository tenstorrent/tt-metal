# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.demos.llama31_8b_p100_v3.tt import eagle_head
from models.ttt_compat.tt.common import PagedAttentionConfig
from models.ttt_compat.tt.generator import Generator, SamplingParams
from models.ttt_compat.tt.model_config import DecodersPrecision

STEPS = 32


def _rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _rope(pos, dim, base):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    fr = torch.outer(pos.float(), inv)
    e = torch.cat([fr, fr], -1)
    return e.cos(), e.sin()


def _rot(x, c, s):
    h = x.shape[-1] // 2
    return x * c + torch.cat([-x[..., h:], x[..., :h]], -1) * s


class Reference:
    def __init__(self, weights, config):
        self.w = {k: (v.float() if v.is_floating_point() else v) for k, v in weights.items()}
        self.heads, self.kv_heads, self.dim = (
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["hidden_size"],
        )
        self.head_dim, self.eps = self.dim // self.heads, config["rms_norm_eps"]
        self.base = float(config.get("rope_theta", 10000.0))
        self.d2t = weights["d2t"].long()
        self.k = torch.zeros(0, self.kv_heads * self.head_dim)
        self.v = torch.zeros(0, self.kv_heads * self.head_dim)

    def step(self, hidden, embedding, position):
        w = self.w
        x = torch.cat(
            [
                _rms(embedding, w["midlayer.input_layernorm.weight"], self.eps),
                _rms(hidden, w["midlayer.hidden_norm.weight"], self.eps),
            ],
            -1,
        )
        q = x @ w["midlayer.self_attn.q_proj.weight"].T
        self.k = torch.cat([self.k, (x @ w["midlayer.self_attn.k_proj.weight"].T)[None]])
        self.v = torch.cat([self.v, (x @ w["midlayer.self_attn.v_proj.weight"].T)[None]])
        n = self.k.shape[0]
        cq, sq = _rope(torch.tensor([position]), self.head_dim, self.base)
        ck, sk = _rope(torch.arange(n), self.head_dim, self.base)
        groups = self.heads // self.kv_heads
        qh = _rot(q.view(1, self.heads, self.head_dim).transpose(0, 1), cq, sq)
        kh = _rot(self.k.view(n, self.kv_heads, self.head_dim).transpose(0, 1), ck, sk).repeat_interleave(groups, 0)
        vh = self.v.view(n, self.kv_heads, self.head_dim).transpose(0, 1).repeat_interleave(groups, 0)
        attn = ((qh @ kh.transpose(-1, -2)) / self.head_dim**0.5).softmax(-1) @ vh
        h = hidden.float() + attn.transpose(0, 1).reshape(self.dim) @ w["midlayer.self_attn.o_proj.weight"].T
        m = _rms(h, w["midlayer.post_attention_layernorm.weight"], self.eps)
        gate = torch.nn.functional.silu(m @ w["midlayer.mlp.gate_proj.weight"].T)
        h = h + (gate * (m @ w["midlayer.mlp.up_proj.weight"].T)) @ w["midlayer.mlp.down_proj.weight"].T
        logits = _rms(h, w["norm.weight"], self.eps) @ w["lm_head.weight"].T
        d = int(logits.argmax(-1))
        return h, d + int(self.d2t[d])


def _pcc(a, b):
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def test_missing_checkpoint(tmp_path, expect_error):
    with expect_error(FileNotFoundError, "draft head: .* not found"):
        eagle_head.read_checkpoint(str(tmp_path))


def test_vocab_order(expect_error):
    d2t = torch.tensor([0, 1, 1, 3])
    order, inverse, count = eagle_head.vocab_order(d2t, 100)
    assert count == 4
    assert order[:4].tolist() == [0, 2, 3, 6]
    assert sorted(order.tolist()) == list(range(100))
    assert all(int(inverse[int(order[i])]) == i for i in range(100))
    with expect_error(ValueError, "token map does not fit"):
        eagle_head.vocab_order(torch.tensor([0, 200]), 100)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": True, "num_command_queues": 1, "trace_region_size": 150000000}], indirect=True
)
def test_head_step(mesh_device):
    paged = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    args, head, kv, order, inverse, weights, embedding, config = eagle_head.create(
        mesh_device,
        eagle_head.head_dir(),
        os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        os.path.join(os.environ.get("TT_CACHE_PATH", "/tmp"), "eagle_head"),
        lambda a: DecodersPrecision.performance(a.n_layers, a.model_name),
        8192,
        paged,
    )
    generator = Generator([head], [args], mesh_device, tokenizer=args.tokenizer)
    sampling = SamplingParams(temperature=0.0, top_k=32, top_p=0.08)
    page_table = torch.arange(1024, dtype=torch.int32).reshape(1, -1)
    state = {"fresh": True}

    def set_hidden(vector):
        t = torch.zeros(1, 1, 32, vector.shape[-1])
        t[0, 0, 0] = vector.float()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
            ),
            head.hidden,
        )

    def step(token, position):
        out = generator.decode_forward(
            torch.tensor([[int(inverse[int(token)])]], dtype=torch.int64),
            torch.tensor([position], dtype=torch.int64),
            page_table=page_table,
            kv_cache=[kv],
            enable_trace=True,
            sampling_params=sampling,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=state["fresh"],
        )
        state["fresh"] = False
        t = out[0] if isinstance(out, tuple) else out
        return int(order[int(t.reshape(-1)[0])])

    reference = Reference(weights, config)
    g = torch.Generator().manual_seed(1115)
    seed = [int(x) for x in order[torch.randint(0, 32000, (4,), generator=g)]]
    features = torch.cat([embedding[seed[0]], embedding[seed[1]], embedding[seed[2]]]).float() * 40.0
    hidden = features @ weights["fc.weight"].float().T
    token = seed[3]
    chain = []
    for p in range(STEPS):
        h, next_token = reference.step(hidden, embedding[token], p)
        chain.append((hidden, token, h, next_token))
        hidden, token = h, next_token

    for _ in range(2):
        set_hidden(chain[0][0])
        step(chain[0][1], 0)
    matches, correlations = 0, []
    for p, (hidden, token, h, next_token) in enumerate(chain):
        set_hidden(hidden)
        got = step(token, p)
        device_h = ttnn.to_torch(head.hidden, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
            0, 0, 0
        ].float()
        matches += int(got == next_token)
        correlations.append(_pcc(device_h - hidden.float(), h - hidden.float()))
    assert head.qkv.calls > 0
    assert matches == STEPS, f"{matches} of {STEPS} tokens match the reference"
    assert min(correlations) > 0.97, f"hidden-state update correlation {min(correlations):.4f}"
