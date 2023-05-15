"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from pymetal import ttmetal as ttm
from python_api_testing.models.bert.embeddings import PytorchEmbeddings
from python_api_testing.models.bert.mha import TtMultiHeadAttentionModel
from python_api_testing.models.bert.ffn import TtFeedForwardModel
from python_api_testing.models.bert.bert_encoder import TtBertEncoder
#from python_api_testing.models.nanogpt.ref.model import get_ddict
from ref.model import ddict as d # for debugging/ref comparison
from python_api_testing.fused_ops.layernorm import Layernorm as ttLayernorm
from python_api_testing.fused_ops.add_and_norm import AddAndNorm
from python_api_testing.fused_ops.linear import Linear as ttLinear
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax, print_diff_tt_pyt, ttP
from utility_functions import tt2torch, tt2torch_rm
from utility_functions import enable_compile_cache
from utility_functions import disable_compile_cache
from utility_functions import get_compile_cache_enabled
from utility_functions import roundup32
from fused_ops.softmax import softmax

class CausalSelfAttention(nn.Module):

    def __init__(self, config, block_index, state_dict, device):
        super().__init__()
        self.index = block_index
        self.device = device
        assert config.n_embd % config.n_head == 0

        # no biases in default Shakespeare config
        qkv_weight = state_dict[f"transformer.h.{block_index}.attn.c_attn.weight"]
        lin_attn_w_q = tilize_to_list(pad_weight(qkv_weight[0:128,:])) # replaces .split call in ref model
        lin_attn_w_k = tilize_to_list(pad_weight(qkv_weight[128:128*2,:]))
        lin_attn_w_v = tilize_to_list(pad_weight(qkv_weight[128*2:128*3,:]))
        lin_proj_w = tilize_to_list(pad_weight(state_dict[f"transformer.h.{block_index}.attn.c_proj.weight"]))

        # replace the original emb,3emb matrix with 3 matrices
        self.c_attn_q = ttLinear(config.n_embd, config.n_embd, lin_attn_w_q, None, device)
        self.c_attn_k = ttLinear(config.n_embd, config.n_embd, lin_attn_w_k, None, device)
        self.c_attn_v = ttLinear(config.n_embd, config.n_embd, lin_attn_w_v, None, device)
        self.c_proj = ttLinear(config.n_embd, config.n_embd, lin_proj_w, None, device)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            torch_mask = torch.tril(torch.ones(config.block_size, config.block_size)) \
                              .view(1, 1, config.block_size, config.block_size)
            self.register_buffer("bias", torch_mask)
            self.tt_mask = tilize_to_list(pad_weight(torch_mask))

        assert(self.n_embd % self.n_head == 0)
        # Used to scale down the input to the softmax
        self.reciprocal_of_sqrt_hidden_dim_tensor = ttm.tensor.Tensor(
            [1 / math.sqrt(self.n_embd/self.n_head)] + [0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.TILE,
            device
        )

    # (N,1,s,e)->(N,s,h,e/h)->(N,h,s,e/h)
    def make_heads(self, x, num_heads):
        untilized_x = ttm.tensor.untilize(x)
        reshaped_unt = ttm.tensor.reshape(untilized_x, x.shape()[0], x.shape()[2], num_heads, x.shape()[3] // num_heads)
        transposed = ttm.tensor.transpose_hc_rm(reshaped_unt)
        retilized = ttm.tensor.tilize(transposed)
        return retilized

    def unmake_heads(self, x):
        untilized_x = ttm.tensor.untilize(x)
        ctx = ttm.tensor.transpose_hc_rm(untilized_x)
        ushape = ctx.shape()
        reshaped = ttm.tensor.reshape(ctx, 1, ushape[0], ushape[1], ushape[2]*ushape[3])
        #set_FR(1)
        retval = ttm.tensor.tilize(reshaped)
        return retval

    def multiply_by_sqrt_hidden_dim(self, x):
        return ttm.tensor.bcast(
            x,
            self.reciprocal_of_sqrt_hidden_dim_tensor,
            ttm.tensor.BcastOpMath.MUL,
            ttm.tensor.BcastOpDim.HW
        )

    def forward(self, x, seq):
        xshp = x.shape()
        B, T, C = xshp[0], xshp[2], xshp[3] # batch size, sequence length, embedding dimensionality (n_embd)
        si = str(self.index)
        #print_diff_tt_pyt(x, d["ax"+si].unsqueeze(1))

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_attn_q(x)
        k = self.c_attn_k(x)
        v = self.c_attn_v(x)
        #print_diff_tt_pyt(q, d['q0'+si].unsqueeze(1))
        #print_diff_tt_pyt(k, d['k0'+si].unsqueeze(1))
        #print_diff_tt_pyt(v, d['v0'+si].unsqueeze(1))
        qh = self.make_heads(q, self.n_head)
        kh = self.make_heads(k, self.n_head)
        vh = self.make_heads(v, self.n_head)
        # make_heads: (N,1,s,e)->(N,s,h,e/h)->(N,h,s,e/h)
        kth = ttm.tensor.transpose(kh) # (N,h,s,e/h) -> (N,h,e/h,s)
        qkt = ttm.tensor.bmm(qh, kth) # e/h is contracted -> (N,h,s,s)

        # N,C,H,W <-> batch, heads, sequence, emb/head
        N, C, H, W = qkt.shape()
        assert(H==W) # qkt should have a shape of (N,h,s,s)
        attention_score_input = self.multiply_by_sqrt_hidden_dim(qkt)

        # create a mask out the sequences to a multiple of 32 (at this point W-T)
        padded_seq_masku = ttm.tensor.fill_rm(N, C, H, W, seq, seq, x, 0x0, 0xc7c3) # 0.0 and -100000 in bf16
        padded_seq_mask = ttm.tensor.tilize(padded_seq_masku)

        # add -100000 to mask out the scores
        attention_score_input_masked = ttm.tensor.add(attention_score_input, padded_seq_mask)

        attention_scores = softmax(attention_score_input_masked)

        # Apply attention to value matrix
        weighted_activation = ttm.tensor.bmm(attention_scores, vh)
        unmade_heads = self.unmake_heads(weighted_activation) # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]
        retval = self.c_proj(unmade_heads)
        #print_diff_tt_pyt(retval, d['atty'+si].unsqueeze(1))
        return retval

class MLP(nn.Module):

    def __init__(self, config, block_index, state_dict, device):
        super().__init__()
        self.c_fc_old    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj_old  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout     = nn.Dropout(config.dropout)
        self.si = str(block_index)

        # no bias in default Shakespeare config
        fc_w        = tilize_to_list(pad_weight(state_dict[f"transformer.h.{block_index}.mlp.c_fc.weight"]))
        proj_w      = tilize_to_list(pad_weight(state_dict[f"transformer.h.{block_index}.mlp.c_proj.weight"]))
        self.c_fc   = ttLinear(config.n_embd, 4*config.n_embd, fc_w, None, device)
        self.c_proj = ttLinear(4*config.n_embd, config.n_embd, proj_w, None, device)

    def forward(self, x):
        x1 = self.c_fc(x)
        #print_diff_tt_pyt(x1, d["mlp_cfc"+self.si])
        x2 = ttm.tensor.gelu(x1) # was: new_gelu
        #print_diff_tt_pyt(x2, d["mlp_gelu"+self.si])
        x3 = self.c_proj(x2)
        #print_diff_tt_pyt(x3, d["mlp_cproj"+self.si])
        #x4 = self.dropout(x3)
        return x3

class FuncModuleWrapper(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x, hh, x1, x2):
        result = self.func(x, hh, x1, x2)
        return result

class Block(nn.Module):

    def __init__(self, config, block_index, state_dict, device):
        super().__init__()
        ln1_dict = state_dict[f"transformer.h.{block_index}.ln_1.weight"]
        ln2_dict = state_dict[f"transformer.h.{block_index}.ln_2.weight"]
        ln1_gamma = tilize_to_list(pad_weight(ln1_dict))
        ln2_gamma = tilize_to_list(pad_weight(ln2_dict))
        self.debug_gamma = ln1_dict
        self.sbi = str(block_index)

        # here we need to pass the "true" shape for reduce to match with zero-padding
        self.ln_1 = ttLayernorm(ln1_gamma, None, 1e-5, 1, ln1_dict.shape[0], device, num_dims=1)
        self.ln_2 = ttLayernorm(ln2_gamma, None, 1e-5, 1, ln2_dict.shape[0], device, num_dims=1)

        self.attn = CausalSelfAttention(config, block_index, state_dict, device)

        self.mlp = MLP(config, block_index, state_dict, device)

    def forward(self, x, seq):
        #print_diff_tt_pyt(x, d["blockx"+self.sbi].unsqueeze(1), "LN1 in"+self.sbi)
        ln1 = self.ln_1(x, seq) # , d["blockx"+self.sbi], self.debug_gamma)
        #print_diff_tt_pyt(ln1, d["ln1x"+self.sbi].unsqueeze(1), "LN1 out"+self.sbi)
        att1 = self.attn(ln1, seq)
        x1 = ttm.tensor.add(x, att1)

        #print_diff_tt_pyt(x1, d["ln2x"+self.sbi].unsqueeze(1), "LN2 in"+self.sbi)
        ln2 = self.ln_2(x1, seq)
        #print_diff_tt_pyt(ln2, d["ln2y"+self.sbi].unsqueeze(1), "LN2 out"+self.sbi)
        mlp = self.mlp(ln2)
        #print_diff_tt_pyt(mlp, d["mlpy"+self.sbi].unsqueeze(1), "MLP out"+self.sbi)
        x2 = ttm.tensor.add(x1, mlp)
        return x2


class TTGPT(nn.Module):

    def __init__(self, config, state_dict, device, wte_, wpe_):
        super().__init__()
        self.device = device
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.lnf_gamma = state_dict[f"transformer.ln_f.weight"]
        lnf_gamma = tilize_to_list(pad_weight(self.lnf_gamma))
        lnf = ttLayernorm(lnf_gamma, None, 1e-5, 1, self.lnf_gamma.shape[0], device, num_dims=1)

        # reuse the same embeddings as reference model for now
        self.transformer = nn.ModuleDict(dict(
            wte = wte_, #nn.Embedding(config.vocab_size, config.n_embd),
            wpe = wpe_, #nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i, state_dict, device) for i in range(config.n_layer)]),
            ln_f = FuncModuleWrapper(lnf)
        ))

        self.lm_head_old = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        pyt_lmh_w = state_dict[f"lm_head.weight"]
        padded_lmh_w = pad_weight(pyt_lmh_w)
        lmh_w = tilize_to_list(padded_lmh_w)
        #self.lm_head_old = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head = ttLinear(config.n_embd, roundup32(config.vocab_size), lmh_w, None, device)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = torch.nn.Parameter(pyt_lmh_w) # https://paperswithcode.com/method/weight-tying

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        seqlen = t
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # convert x from pytorch to tt 4D tensor
        x1 = pad_weight(x.unsqueeze(1))
        x_tilized = tilize_to_list(x1)
        x_tt = ttm.tensor.Tensor(
            x_tilized, x1.shape, ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, self.device)
        #print_diff_tt_pyt(x_tt, d["tokpos"].unsqueeze(1), "Embeddings")
        for block in self.transformer.h:
            x_tt = block(x_tt, t)
        #print_diff_tt_pyt(x_tt, d["allblocks"].unsqueeze(1), "allblocks out")
        x_tt1 = self.transformer.ln_f(x_tt, seqlen, d["allblocks"], self.lnf_gamma) # H_
        #print_diff_tt_pyt(x_tt1, d["lnf"].unsqueeze(1), "lnf out")

        if targets is not None:
            assert(False and "Unimplemented.")
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            #logits = self.lm_head(x_tt[:, [-1], :]) # note: using list [-1] to preserve the time dim
            tt_logits = self.lm_head(x_tt1) # note: using list [-1] to preserve the time dim
            #print_diff_tt_pyt(tt_logits, d["logits"], "lmhead out")
            loss = None

        logits_tt_host = tt_logits.to(ttm.device.GetHost())
        logits_tt_py = untilize(torch.Tensor(logits_tt_host.data()).reshape(x.shape[0], 1, x1.shape[2], -1))
        logits_tt_py1 = logits_tt_py[:, :, seqlen-1, 0:self.config.vocab_size] # seqlen-1 corresponds to :-1: slice

        return logits_tt_py1, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, wte=None, wpe=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        #enable_compile_cache()
        torch.manual_seed(2) # must match the seed in ref model.py for same results, due to multinomial()
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            #print_diff_argmax(logits, d["logitsself"])
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            #print_diff_argmax(probs, d["multi_probs"])
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
