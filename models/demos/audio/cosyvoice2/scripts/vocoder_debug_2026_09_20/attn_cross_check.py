"""Decisive empirical test: substitute the REAL diffusers.Attention class (with
real loaded weights) for our hand-rolled BasicTransformerBlockRef attention math,
on real intermediate activations from a real forward pass. If they match, our
attention implementation is cleared and the estimator bug is elsewhere. If they
diverge, this is (part of) the root cause of the ONNX cross-check mismatch
(PCC 0.944 against flow.decoder.estimator.fp32.onnx).
"""
import sys

import torch

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad")

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

print("=== loading real flow.pt + reference audio (same recipe as onnx_cross_check.py) ===")
flow_sd = load_checkpoint_file("flow.pt")

from huggingface_hub import hf_hub_download

campplus_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="campplus.onnx")
st_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="speech_tokenizer_v2.onnx")

import onnxruntime as ort

campplus_session = ort.InferenceSession(campplus_path, providers=["CPUExecutionProvider"])
st_session = ort.InferenceSession(st_path, providers=["CPUExecutionProvider"])

import torchaudio
from datasets import load_dataset

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
ref_ex = ds[0]
ref_wav16 = torch.tensor(ref_ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
ref_wav24 = torchaudio.functional.resample(ref_wav16, 16000, 24000)
prompt_tokens = extract_speech_tokens(st_session, ref_wav16)
prompt_feat = extract_prompt_feat(ref_wav24)
ref_embedding = extract_spk_embedding(campplus_session, ref_wav16)
token_len = min(prompt_feat.shape[1] // 2, prompt_tokens.shape[1])
prompt_feat = prompt_feat[:, : 2 * token_len]
prompt_tokens = prompt_tokens[:, :token_len]

from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef

flow_ref = CausalMaskedDiffWithXvecRef.from_checkpoint(flow_sd)
flow_ref.eval()

torch.manual_seed(123)
new_tokens = torch.randint(0, 6561, (1, 80))

print("=== building the SAME real (x, mask, mu, t, spks, cond) as onnx_cross_check.py ===")
with torch.no_grad():
    spks = torch.nn.functional.normalize(ref_embedding, dim=1)
    spks = flow_ref.spk_embed_affine_layer(spks)

    full_token = torch.cat([prompt_tokens, new_tokens], dim=1)
    tok_emb = flow_ref.input_embedding(full_token.clamp(min=0))
    h = flow_ref.encoder(tok_emb)
    mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
    h = flow_ref.encoder_proj(h)

    conds = torch.zeros(1, mel_len1 + mel_len2, flow_ref.output_size, dtype=h.dtype)
    conds[:, :mel_len1] = prompt_feat

    mu_cl = h  # [1, T, 80] channels-last (this port's own convention)
    T = mel_len1 + mel_len2
    mask_cl = torch.ones(1, T, 1, dtype=h.dtype)

    cfm = flow_ref.decoder
    torch.manual_seed(0)
    z_cl = cfm.rand_noise[:, :T, :]  # [1, T, 80] channels-last
    t_span = torch.linspace(0, 1, 10 + 1)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t0 = t_span[0].unsqueeze(0)

    estimator = flow_ref.decoder.estimator  # CausalConditionalDecoderRef

    # Reproduce estimator.forward()'s OWN preamble exactly, up to the first
    # down_tbs[0] call, so we get the real `h` (post down_resnet, pre-transformer)
    # and real `attn_bias` that down_tbs[0] actually receives mid-inference.
    from models.demos.audio.cosyvoice2.tt.flow.decoder import sinusoidal_pos_emb_torch

    temb = sinusoidal_pos_emb_torch(t0, estimator.time_embeddings_dim)
    temb = estimator.time_mlp(temb)
    hcat = torch.cat([z_cl, mu_cl], dim=-1)
    hcat = torch.cat([hcat, spks.unsqueeze(1).expand(-1, hcat.shape[1], -1)], dim=-1)
    hcat = torch.cat([hcat, conds], dim=-1)
    attn_bias = (1.0 - mask_cl.transpose(1, 2)).float() * -1.0e10
    attn_bias = attn_bias.unsqueeze(1)
    hh = hcat.transpose(1, 2)
    mask_cf = mask_cl.transpose(1, 2)
    hh = estimator.down_resnet(hh, mask_cf, temb)
    hh = hh.transpose(1, 2)  # [1, T, 256] -- this is the REAL input down_tbs[0] sees

print(f"real transformer-block input hh: shape={hh.shape} mean={hh.mean().item():.5f} std={hh.std().item():.5f}")
print(f"real attn_bias: shape={attn_bias.shape}")

block = estimator.down_tbs[0]  # BasicTransformerBlockRef, real loaded weights

print("\n=== our own BasicTransformerBlockRef attention math ===")
with torch.no_grad():
    hn = block.norm1(hh)
    b, t, _ = hn.shape

    def _heads(m):
        return m.reshape(b, t, block.num_heads, block.head_dim).transpose(1, 2)

    q, k, v = _heads(block.to_q(hn)), _heads(block.to_k(hn)), _heads(block.to_v(hn))
    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    attn = attn.transpose(1, 2).reshape(b, t, block.num_heads * block.head_dim)
    ours_out = block.to_out(attn)
print(f"ours: shape={ours_out.shape} mean={ours_out.mean().item():.6f} std={ours_out.std().item():.6f}")

print("\n=== REAL diffusers.Attention, same real weights, same real input ===")
from diffusers.models.attention_processor import Attention

real_attn = Attention(query_dim=256, heads=block.num_heads, dim_head=block.head_dim, dropout=0.0, bias=False)
with torch.no_grad():
    real_attn.to_q.weight.copy_(block.to_q.weight)
    real_attn.to_k.weight.copy_(block.to_k.weight)
    real_attn.to_v.weight.copy_(block.to_v.weight)
    real_attn.to_out[0].weight.copy_(block.to_out.weight)
    real_attn.to_out[0].bias.copy_(block.to_out.bias)
real_attn.eval()

with torch.no_grad():
    hn2 = block.norm1(hh)  # identical norm1 call, same weights, deterministic
    diffusers_out = real_attn(hn2, attention_mask=attn_bias)
print(f"diffusers: shape={diffusers_out.shape} mean={diffusers_out.mean().item():.6f} std={diffusers_out.std().item():.6f}")

print("\n=== COMPARISON (ours vs. real diffusers.Attention, identical weights+input) ===")
from models.common.utility_functions import comp_pcc

passed, pcc = comp_pcc(diffusers_out, ours_out, 0.999)
max_diff = (diffusers_out - ours_out).abs().max().item()
mean_diff = (diffusers_out - ours_out).abs().mean().item()
print(f"PCC: {pcc}")
print(f"max abs diff: {max_diff:.8f}")
print(f"mean abs diff: {mean_diff:.8f}")
print("PASS -- attention math is NOT the bug" if passed else "MISMATCH -- attention math IS (part of) the bug")
