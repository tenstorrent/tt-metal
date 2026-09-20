"""Isolate the FFN half of BasicTransformerBlockRef (norm3 -> ff_in -> gelu ->
ff_out) the same way attention was isolated: substitute the real
diffusers.models.attention.FeedForward class (activation_fn="gelu"), with the
SAME real loaded weights, fed the SAME real activation entering down_tbs[0],
and compare against our own implementation bit-for-bit.
"""
import sys

import torch

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad")

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

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

with torch.no_grad():
    spks = torch.nn.functional.normalize(ref_embedding, dim=1)
    spks = flow_ref.spk_embed_affine_layer(spks)
    full_token = torch.cat([prompt_tokens, new_tokens], dim=1)
    tok_emb = flow_ref.input_embedding(full_token.clamp(min=0))
    h_enc = flow_ref.encoder(tok_emb)
    mel_len1, mel_len2 = prompt_feat.shape[1], h_enc.shape[1] - prompt_feat.shape[1]
    h_enc = flow_ref.encoder_proj(h_enc)
    conds = torch.zeros(1, mel_len1 + mel_len2, flow_ref.output_size, dtype=h_enc.dtype)
    conds[:, :mel_len1] = prompt_feat
    mu_cl = h_enc
    T = mel_len1 + mel_len2
    mask_cl = torch.ones(1, T, 1, dtype=h_enc.dtype)

    cfm = flow_ref.decoder
    torch.manual_seed(0)
    z_cl = cfm.rand_noise[:, :T, :]
    t_span = torch.linspace(0, 1, 10 + 1)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t0 = t_span[0].unsqueeze(0)

    b = 1
    x_in_cl = torch.zeros(2 * b, T, 80)
    mask_in_cl = torch.zeros(2 * b, T, 1)
    mu_in_cl = torch.zeros(2 * b, T, 80)
    t_in = torch.zeros(2 * b)
    spks_in = torch.zeros(2 * b, 80)
    cond_in_cl = torch.zeros(2 * b, T, 80)
    x_in_cl[:] = z_cl
    mask_in_cl[:] = mask_cl
    mu_in_cl[:b] = mu_cl
    t_in[:] = t0
    spks_in[:b] = spks
    cond_in_cl[:b] = conds

from models.demos.audio.cosyvoice2.tt.flow.decoder import sinusoidal_pos_emb_torch

estimator = flow_ref.decoder.estimator
with torch.no_grad():
    temb = sinusoidal_pos_emb_torch(t_in, estimator.time_embeddings_dim)
    temb = estimator.time_mlp(temb)
    hh = torch.cat([x_in_cl, mu_in_cl], dim=-1)
    hh = torch.cat([hh, spks_in.unsqueeze(1).expand(-1, hh.shape[1], -1)], dim=-1)
    hh = torch.cat([hh, cond_in_cl], dim=-1)
    attn_bias = (1.0 - mask_in_cl.transpose(1, 2)).float() * -1.0e10
    attn_bias = attn_bias.unsqueeze(1)
    hh = hh.transpose(1, 2)
    mask_cf = mask_in_cl.transpose(1, 2)
    hh = estimator.down_resnet(hh, mask_cf, temb)
    hh = hh.transpose(1, 2)  # real input entering down_tbs[0]

block = estimator.down_tbs[0]

print("=== our own full block forward (attention half + FFN half), real input/weights ===")
with torch.no_grad():
    b_, t_, _ = hh.shape
    h1 = block.norm1(hh)

    def _heads(m):
        return m.reshape(b_, t_, block.num_heads, block.head_dim).transpose(1, 2)

    q, k, v = _heads(block.to_q(h1)), _heads(block.to_k(h1)), _heads(block.to_v(h1))
    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    attn = attn.transpose(1, 2).reshape(b_, t_, block.num_heads * block.head_dim)
    x_pre_ffn = hh + block.to_out(attn)

    h3 = block.norm3(x_pre_ffn)
    ffn_out_ours = block.ff_out(torch.nn.functional.gelu(block.ff_in(h3)))
    full_out_ours = x_pre_ffn + ffn_out_ours

print(f"x_pre_ffn: mean={x_pre_ffn.mean().item():.6f} std={x_pre_ffn.std().item():.6f}")
print(f"h3 (post norm3): mean={h3.mean().item():.6f} std={h3.std().item():.6f}")
print(f"ffn_out_ours: mean={ffn_out_ours.mean().item():.6f} std={ffn_out_ours.std().item():.6f}")

print("\n=== real diffusers.FeedForward, same real weights, same real h3 input ===")
from diffusers.models.attention import FeedForward

real_ff = FeedForward(dim=256, activation_fn="gelu", mult=4, dropout=0.0, bias=True)
with torch.no_grad():
    # net[0] is GELU(dim,inner_dim) wrapping .proj; net[1] dropout (no-op eval); net[2] Linear(inner_dim,dim)
    real_ff.net[0].proj.weight.copy_(block.ff_in.weight)
    real_ff.net[0].proj.bias.copy_(block.ff_in.bias)
    real_ff.net[2].weight.copy_(block.ff_out.weight)
    real_ff.net[2].bias.copy_(block.ff_out.bias)
real_ff.eval()

with torch.no_grad():
    ffn_out_diffusers = real_ff(h3)

print(f"ffn_out_diffusers: mean={ffn_out_diffusers.mean().item():.6f} std={ffn_out_diffusers.std().item():.6f}")

from models.common.utility_functions import comp_pcc

print("\n=== COMPARISON: FFN output only (ours vs real diffusers.FeedForward) ===")
_, pcc = comp_pcc(ffn_out_diffusers, ffn_out_ours, 0.999)
max_diff = (ffn_out_diffusers - ffn_out_ours).abs().max().item()
print(f"PCC={pcc}  max_abs_diff={max_diff:.8f}")
print("PASS -- FFN math matches" if max_diff < 1e-4 else "MISMATCH -- FFN math diverges")

print("\n=== sanity: does our own full-block output match the earlier stage_bisect.py capture? ===")
print(f"full_out_ours: mean={full_out_ours.mean().item():.6f} std={full_out_ours.std().item():.6f} (expect std~0.6899, from down_blocks.0.1.0 row)")
