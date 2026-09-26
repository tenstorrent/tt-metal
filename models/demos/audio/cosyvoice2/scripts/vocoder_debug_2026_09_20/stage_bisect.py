"""Single-call stage-by-stage bisection: same real (x, mask, mu, t, spks, cond)
that produced PCC 0.944 / max_diff 5.99 on the full estimator output, now
compared at every stage boundary inside CausalConditionalDecoderRef.forward
against the ONNX graph's own real intermediate tensors (extracted by adding
them as extra graph outputs -- node names carry the real module scope, e.g.
'/down_blocks.0.0/Add_1_output_0', confirmed against flow.pt's own key names).
"""
import sys

import numpy as np
import onnx
import torch

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad")

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file

print("=== loading real flow.pt + reference audio ===")
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

print("=== building the SAME real (x, mask, mu, t, spks, cond) as onnx_cross_check.py (step-1 call) ===")
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

    mu_cl = h_enc  # [1, T, 80]
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

print("\n=== extracting ONNX intermediate tensors (add as extra graph outputs) ===")
STAGE_NAMES = [
    "/down_blocks.0.0/Add_1_output_0",
    "/down_blocks.0.1.0/Add_1_output_0",
    "/down_blocks.0.1.1/Add_1_output_0",
    "/down_blocks.0.1.2/Add_1_output_0",
    "/down_blocks.0.1.3/Add_1_output_0",
    "/down_blocks.0.2/Conv_output_0",
    "/mid_blocks.0.0/Add_1_output_0",
    "/mid_blocks.0.1.3/Add_1_output_0",
    "/mid_blocks.5.0/Add_1_output_0",
    "/mid_blocks.5.1.3/Add_1_output_0",
    "/mid_blocks.11.0/Add_1_output_0",
    "/mid_blocks.11.1.3/Add_1_output_0",
    "/up_blocks.0.0/Add_1_output_0",
    "/up_blocks.0.1.3/Add_1_output_0",
    "/up_blocks.0.2/Conv_output_0",
    "/final_block/Mul_1_output_0",
    "/final_proj/Conv_output_0",
]

model = onnx.load(onnx_path)
existing_out_names = {o.name for o in model.graph.output}
for name in STAGE_NAMES:
    if name not in existing_out_names:
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
all_out_names = [o.name for o in sess.get_outputs()]
onnx_results = sess.run(
    all_out_names,
    {"x": x_in_cf, "mask": mask_in_cf, "mu": mu_in_cf, "t": t_in_np, "spks": spks_in_np, "cond": cond_in_cf},
)
onnx_by_name = dict(zip(all_out_names, onnx_results))
print(f"got {len(onnx_by_name)} onnx outputs, including final 'estimator_out'")

print("\n=== running our torch reference estimator, capturing the SAME stage boundaries ===")
from models.demos.audio.cosyvoice2.tt.flow.decoder import sinusoidal_pos_emb_torch

estimator = flow_ref.decoder.estimator  # CausalConditionalDecoderRef, real loaded weights
torch_stages = {}
with torch.no_grad():
    temb = sinusoidal_pos_emb_torch(t_in, estimator.time_embeddings_dim)
    temb = estimator.time_mlp(temb)

    hh = torch.cat([x_in_cl, mu_in_cl], dim=-1)
    hh = torch.cat([hh, spks_in.unsqueeze(1).expand(-1, hh.shape[1], -1)], dim=-1)
    hh = torch.cat([hh, cond_in_cl], dim=-1)

    attn_bias = (1.0 - mask_in_cl.transpose(1, 2)).float() * -1.0e10
    attn_bias = attn_bias.unsqueeze(1)

    hh = hh.transpose(1, 2)  # channel-first
    mask_cf = mask_in_cl.transpose(1, 2)

    hh = estimator.down_resnet(hh, mask_cf, temb)
    torch_stages["/down_blocks.0.0/Add_1_output_0"] = hh.clone()

    hh = hh.transpose(1, 2)
    for tb_i, tb in enumerate(estimator.down_tbs):
        hh = tb(hh, attn_bias)
        torch_stages[f"/down_blocks.0.1.{tb_i}/Add_1_output_0"] = hh.clone()  # channel-last, matches ONNX here
    skip = hh

    hh = hh.transpose(1, 2)
    hh = estimator.down_conv(hh * mask_cf)
    torch_stages["/down_blocks.0.2/Conv_output_0"] = hh.clone()

    mid_capture_idx = {0: "0", 5: "5", 11: "11"}
    for i, (resnet, tbs) in enumerate(zip(estimator.mid_resnets, estimator.mid_tbs)):
        hh = resnet(hh, mask_cf, temb)
        if i in mid_capture_idx:
            torch_stages[f"/mid_blocks.{mid_capture_idx[i]}.0/Add_1_output_0"] = hh.clone()
        hh = hh.transpose(1, 2)
        for tb in tbs:
            hh = tb(hh, attn_bias)
        if i in mid_capture_idx:
            torch_stages[f"/mid_blocks.{mid_capture_idx[i]}.1.3/Add_1_output_0"] = hh.clone()  # channel-last
        hh = hh.transpose(1, 2)

    hh = hh.transpose(1, 2)
    hh = torch.cat([hh, skip], dim=-1)
    hh = hh.transpose(1, 2)
    hh = estimator.up_resnet(hh, mask_cf, temb)
    torch_stages["/up_blocks.0.0/Add_1_output_0"] = hh.clone()

    hh = hh.transpose(1, 2)
    for tb in estimator.up_tbs:
        hh = tb(hh, attn_bias)
    torch_stages["/up_blocks.0.1.3/Add_1_output_0"] = hh.clone()  # channel-last

    hh = hh.transpose(1, 2)
    hh = estimator.up_conv(hh * mask_cf)
    torch_stages["/up_blocks.0.2/Conv_output_0"] = hh.clone()

    hh = estimator.final_block(hh, mask_cf)
    torch_stages["/final_block/Mul_1_output_0"] = hh.clone()

    out = estimator.final_proj(hh * mask_cf)
    torch_stages["/final_proj/Conv_output_0"] = out.clone()

print("\n=== COMPARISON TABLE (channel-first [2,C,T] both sides) ===")
from models.common.utility_functions import comp_pcc

order = STAGE_NAMES
print(f"{'stage':35} | {'PCC':>10} | {'max_diff':>10} | {'mean_diff':>10} | {'ours_std':>9} | {'onnx_std':>9}")
for name in order:
    ours_t = torch_stages[name]
    onnx_t = torch.from_numpy(np.asarray(onnx_by_name[name]))
    if ours_t.shape != onnx_t.shape:
        print(f"{name:35} | SHAPE MISMATCH ours={tuple(ours_t.shape)} onnx={tuple(onnx_t.shape)}")
        continue
    _, pcc = comp_pcc(onnx_t, ours_t, 0.99)
    max_diff = (onnx_t - ours_t).abs().max().item()
    mean_diff = (onnx_t - ours_t).abs().mean().item()
    print(
        f"{name:35} | {pcc:>10.6f} | {max_diff:>10.5f} | {mean_diff:>10.6f} | "
        f"{ours_t.std().item():>9.5f} | {onnx_t.std().item():>9.5f}"
    )

print("\n=== full estimator_out (sanity check against known PCC 0.944) ===")
final_onnx = torch.from_numpy(np.asarray(onnx_by_name["estimator_out"]))
final_ours = torch_stages["/final_proj/Conv_output_0"]
_, pcc = comp_pcc(final_onnx, final_ours, 0.99)
print(f"estimator_out vs final_proj (should match earlier 0.944 finding): PCC={pcc}, max_diff={(final_onnx-final_ours).abs().max().item():.5f}")
