# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os

import torch

import ttnn
from models.demos.llama31_8b_p100_spec.tt.model import Transformer
from models.demos.llama31_8b_p100_spec.tt.spec_decode import draft_config
from models.ttt_compat.tt.common import Mode
from models.ttt_compat.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys
from models.ttt_compat.tt.model_config import ModelArgs, TensorGroup

PAD_ROWS = draft_config("eagle")["pad_rows"]
FILES = ("config.json", "model.safetensors")
HEAD_ID = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


def head_dir():
    return os.environ.get("HF_EAGLE_HEAD", HEAD_ID)


def local_dir(path):
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(path, allow_patterns=["*.json", "*.safetensors"])


def read_checkpoint(head_dir):
    from safetensors import safe_open

    head_dir = local_dir(head_dir)
    for f in FILES:
        if not os.path.isfile(os.path.join(head_dir, f)):
            raise FileNotFoundError(f"draft head: {f} not found in {head_dir}")
    with safe_open(os.path.join(head_dir, "model.safetensors"), "pt") as h:
        weights = {k: h.get_tensor(k) for k in h.keys()}
    with open(os.path.join(head_dir, "config.json")) as f:
        config = json.load(f)
    return weights, config


def read_embedding(model_dir):
    from safetensors import safe_open

    model_dir = local_dir(model_dir)
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, f), "pt") as h:
                if "model.embed_tokens.weight" in h.keys():
                    return h.get_tensor("model.embed_tokens.weight")
    raise FileNotFoundError(f"model.embed_tokens.weight not found in {model_dir}")


def vocab_order(d2t, vocab):
    d2t = d2t.long()
    head_tokens = (torch.arange(d2t.shape[0]) + d2t).tolist()
    if len(set(head_tokens)) != len(head_tokens) or max(head_tokens) >= vocab:
        raise ValueError("draft head: token map does not fit the model vocabulary")
    used = set(head_tokens)
    filler = [i for i in range(vocab) if i not in used][:PAD_ROWS]
    first = torch.tensor(head_tokens + filler, dtype=torch.int64)
    rest = torch.ones(vocab, dtype=torch.bool)
    rest[first] = False
    order = torch.cat([first, torch.arange(vocab)[rest]])
    inverse = torch.empty(vocab, dtype=torch.int64)
    inverse[order] = torch.arange(vocab)
    return order, inverse, len(head_tokens)


def hf_names(weights, embedding, order):
    def bf(t):
        return t.to(torch.bfloat16)

    dim = weights["midlayer.self_attn.o_proj.weight"].shape[0]
    lm = weights["lm_head.weight"]
    sd = {
        "model.embed_tokens.weight": bf(embedding[order]),
        "lm_head.weight": bf(torch.cat([lm, torch.zeros(PAD_ROWS, dim, dtype=lm.dtype)])),
        "model.norm.weight": bf(weights["norm.weight"]),
    }
    halves = ((slice(dim, 2 * dim), "midlayer.hidden_norm.weight"), (slice(0, dim), "midlayer.input_layernorm.weight"))
    for layer, (cols, norm) in enumerate(halves):
        p = f"model.layers.{layer}."
        for n in ("q", "k", "v"):
            sd[p + f"self_attn.{n}_proj.weight"] = bf(
                weights[f"midlayer.self_attn.{n}_proj.weight"][:, cols].contiguous()
            )
        sd[p + "self_attn.o_proj.weight"] = bf(weights["midlayer.self_attn.o_proj.weight"])
        for n in ("gate", "up", "down"):
            sd[p + f"mlp.{n}_proj.weight"] = bf(weights[f"midlayer.mlp.{n}_proj.weight"])
        sd[p + "input_layernorm.weight"] = bf(weights[norm])
        sd[p + "post_attention_layernorm.weight"] = bf(weights["midlayer.post_attention_layernorm.weight"])
    return sd


class HeadArgs(ModelArgs):
    def __init__(self, *args, head_state=None, head_vocab=None, head_theta=10000.0, **kwargs):
        self._head_state, self._head_vocab, self._head_theta = head_state, head_vocab, head_theta
        super().__init__(*args, **kwargs)

    def _set_params_from_dict(self, config):
        r = super()._set_params_from_dict(config)
        self.vocab_size = self.padded_vocab_size = self._head_vocab
        self.rope_theta, self.rope_scaling = self._head_theta, None
        return r

    def load_state_dict(self):
        self.is_mixture_of_experts = False
        self.fuse_qkv = self.fuse_mlp = False
        sd = standardize_hf_keys(dict(self._head_state))
        return convert_hf_to_meta(sd, self.head_dim, self.n_heads, self.n_kv_heads)


class QkvInputs:
    def __init__(self, base, weight, extra_weight):
        self._base, self._weight, self._extra_weight = base, weight, extra_weight
        self.extra, self.calls = None, 0

    def __getattr__(self, name):
        return getattr(self._base, name)

    def linear(self, x, w, *args, **kwargs):
        y = self._base.linear(x, w, *args, **kwargs)
        if w is not self._weight or self.extra is None:
            return y
        z = self._base.linear(self.extra, self._extra_weight, *args, **kwargs)
        out = ttnn.add(y, z, memory_config=y.memory_config())
        ttnn.deallocate(y)
        ttnn.deallocate(z)
        self.calls += 1
        return out


class EagleHead(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_half, embedding_half = self.layers[0].attention, self.layers[1].attention
        if getattr(hidden_half, "_matmul_bh", None) is None:
            raise RuntimeError("draft head: the llama31_8b_spec matmul programs are not available on this device")
        self.qkv = QkvInputs(hidden_half._matmul_bh, hidden_half.wqkv, embedding_half.wqkv)
        hidden_half._matmul_bh = self.qkv
        self.hidden = ttnn.from_torch(
            torch.zeros(1, 1, 32, self.args.dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.row_positions = ()

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs):
        super()._increment_decode_positions_device(current_pos, rot_mat_idxs)
        for t in self.row_positions:
            ttnn.plus_one(t, skip_negative_entries=True)

    def forward(
        self,
        x,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        page_tables_per_layer=None,
    ):
        if mode != Mode.DECODE:
            raise ValueError("draft head: decode only")
        layer, lender = self.layers[0], self.layers[1]
        residual = self.args.get_residual_mem_config(mode, None)
        act = self.args.decoders_optimizations.get_tensor_dtype(decoder_id=0, tensor=TensorGroup.ACTIVATION)
        emb = ttnn.to_memory_config(x, residual, act)
        self.qkv.extra = lender.attention_norm(emb, mode, norm_config=self.args.get_norm_config("attn", mode, None))
        hid = ttnn.to_memory_config(self.hidden, residual, act)
        try:
            h = layer(
                hid,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[0] if kv_cache is not None else None,
                batch_size=batch_size,
            )
        finally:
            self.qkv.extra = None
        hd = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
        ttnn.copy(hd, self.hidden)
        ttnn.deallocate(hd)
        y = self.norm(h, mode=mode, norm_config=self.args.get_norm_config("lm_head", mode, None))
        y = self.lm_head(y)
        return self._apply_final_logit_softcapping(y)


def create(
    mesh_device,
    head_dir,
    model_dir,
    cache_dir,
    optimizations,
    max_seq_len,
    paged_attention_config,
    batch=1,
    dtype=ttnn.bfloat8_b,
    loaded=None,
):
    if loaded is not None:
        weights, embedding, config = loaded
    else:
        weights, config = read_checkpoint(head_dir)
        embedding = read_embedding(model_dir)
    order, inverse, n_head = vocab_order(weights["d2t"], embedding.shape[0])
    state = hf_names(weights, embedding, order)
    previous = os.environ.get("TT_CACHE_PATH")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TT_CACHE_PATH"] = cache_dir
    try:
        args = HeadArgs(
            mesh_device,
            instruct=True,
            max_batch_size=batch,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            head_state=state,
            head_vocab=n_head + PAD_ROWS,
            head_theta=float(config.get("rope_theta", 10000.0)),
        )
        args.n_layers = 2
        state_dict = args.load_state_dict()
        with ttnn.as_tensor_cache_repair(True):
            head = EagleHead(
                args=args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=args.weight_cache_path(dtype),
                paged_attention_config=paged_attention_config,
            )
    finally:
        if previous is None:
            os.environ.pop("TT_CACHE_PATH", None)
        else:
            os.environ["TT_CACHE_PATH"] = previous
    kv = [layer.attention.layer_past for layer in head.layers] if paged_attention_config else None
    return args, head, kv, order, inverse, weights, embedding, config
