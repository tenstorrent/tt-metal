"""Op-by-op bisection inside down_blocks.0.1.0 (first down transformer block).
Both attention and FFN were proven bit-exact against real diffusers classes in
isolation, yet the combined block output still diverges from the ONNX graph's
own node (PCC 0.9989, max_diff 0.404). Extract every op boundary from the ONNX
graph and diff against the identical manual recomputation to find exactly
which single op first introduces a difference.
"""
import sys

import numpy as np
import onnx
import torch

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad")

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

flow_sd = load_checkpoint_file("flow.pt")

from huggingface_hub import hf_hub_download

campplus_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="campplus.onnx")
st_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="speech_tokenizer_v2.onnx")
onnx_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="flow.decoder.estimator.fp32.onnx")

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

x_in_cf = x_in_cl.transpose(1, 2).contiguous().numpy().astype(np.float32)
mask_in_cf = mask_in_cl.transpose(1, 2).contiguous().numpy().astype(np.float32)
mu_in_cf = mu_in_cl.transpose(1, 2).contiguous().numpy().astype(np.float32)
cond_in_cf = cond_in_cl.transpose(1, 2).contiguous().numpy().astype(np.float32)
t_in_np = t_in.numpy().astype(np.float32)
spks_in_np = spks_in.numpy().astype(np.float32)

STAGE_NAMES = [
    "/down_blocks.0.0/Add_1_output_0",
    "/down_blocks.0.1.0/norm1/LayerNormalization_output_0",
    "/down_blocks.0.1.0/attn1/to_out.0/Add_output_0",
    "/down_blocks.0.1.0/attn1/Div_2_output_0",
    "/down_blocks.0.1.0/Add_output_0",
    "/down_blocks.0.1.0/norm3/LayerNormalization_output_0",
    "/down_blocks.0.1.0/ff/net.0/proj/Add_output_0",
    "/down_blocks.0.1.0/ff/net.0/Mul_1_output_0",
    "/down_blocks.0.1.0/ff/net.2/Add_output_0",
    "/down_blocks.0.1.0/Add_1_output_0",
]

model = onnx.load(onnx_path)
existing = {o.name for o in model.graph.output}
for name in STAGE_NAMES:
    if name not in existing:
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
all_out_names = [o.name for o in sess.get_outputs()]
onnx_results = sess.run(
    all_out_names,
    {"x": x_in_cf, "mask": mask_in_cf, "mu": mu_in_cf, "t": t_in_np, "spks": spks_in_np, "cond": cond_in_cf},
)
onnx_by_name = dict(zip(all_out_names, onnx_results))

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
    hh = estimator.down_resnet(hh, mask_cf, temb)  # channel-first, matches down_blocks.0.0

    block_input_cf = hh.clone()
    hh = hh.transpose(1, 2)  # channel-last, entering down_tbs[0]

    block = estimator.down_tbs[0]
    b_, t_, _ = hh.shape
    h1 = block.norm1(hh)

    def _heads(m):
        return m.reshape(b_, t_, block.num_heads, block.head_dim).transpose(1, 2)

    q, k, v = _heads(block.to_q(h1)), _heads(block.to_k(h1)), _heads(block.to_v(h1))
    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    attn = attn.transpose(1, 2).reshape(b_, t_, block.num_heads * block.head_dim)
    to_out_out = block.to_out(attn)  # matches ONNX's to_out.0/Add (pre div-by-1)
    div_out = to_out_out / 1.0  # matches ONNX's Div_2 (rescale_output_factor=1.0)
    x_pre_ffn = hh + div_out  # matches ONNX's post-attention residual Add

    h3 = block.norm3(x_pre_ffn)
    ff_pre_gelu = block.ff_in(h3)  # matches ff/net.0/proj/Add
    ff_post_gelu = torch.nn.functional.gelu(ff_pre_gelu)  # matches ff/net.0/Mul_1
    ffn_out = block.ff_out(ff_post_gelu)  # matches ff/net.2/Add
    full_out = x_pre_ffn + ffn_out  # matches Add_1

torch_by_name = {
    "/down_blocks.0.0/Add_1_output_0": block_input_cf,
    "/down_blocks.0.1.0/norm1/LayerNormalization_output_0": h1,
    "/down_blocks.0.1.0/attn1/to_out.0/Add_output_0": to_out_out,
    "/down_blocks.0.1.0/attn1/Div_2_output_0": div_out,
    "/down_blocks.0.1.0/Add_output_0": x_pre_ffn,
    "/down_blocks.0.1.0/norm3/LayerNormalization_output_0": h3,
    "/down_blocks.0.1.0/ff/net.0/proj/Add_output_0": ff_pre_gelu,
    "/down_blocks.0.1.0/ff/net.0/Mul_1_output_0": ff_post_gelu,
    "/down_blocks.0.1.0/ff/net.2/Add_output_0": ffn_out,
    "/down_blocks.0.1.0/Add_1_output_0": full_out,
}

from models.common.utility_functions import comp_pcc

print(f"{'op':50} | {'PCC':>10} | {'max_diff':>10} | {'shapes match?'}")
for name in STAGE_NAMES:
    ours = torch_by_name[name]
    onnx_t = torch.from_numpy(np.asarray(onnx_by_name[name]))
    same_shape = ours.shape == onnx_t.shape
    if not same_shape and ours.numel() == onnx_t.numel():
        onnx_t = onnx_t.reshape(ours.shape)
        same_shape = True
    if not same_shape:
        print(f"{name:50} | SHAPE MISMATCH ours={tuple(ours.shape)} onnx={tuple(onnx_t.shape)}")
        continue
    _, pcc = comp_pcc(onnx_t, ours, 0.999999)
    max_diff = (onnx_t - ours).abs().max().item()
    print(f"{name:50} | {pcc:>10.8f} | {max_diff:>10.7f} | {same_shape}")
