"""Generate ONE utterance in this process, then exit.

Invoked by run_jp_quality.py; not meant to be run by hand except for debugging.
One generation per process is deliberate: repeated generations in one process
eventually hang in Talker prefill.

The model code is imported from ``--tree`` (any tt-metal checkout or worktree
that has models/demos/qwen3_tts), so the same scripts can test any commit.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--sentences", required=True)
ap.add_argument("--id", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hf-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
ap.add_argument("--max-new-tokens", type=int, default=256)
a = ap.parse_args()

tree = str(Path(a.tree).resolve())
sys.path.insert(0, tree)
import models.demos.qwen3_tts.tt.server as S  # noqa: E402

assert S.__file__.startswith(tree), f"imported {S.__file__}, expected under {tree}"

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import ttnn  # noqa: E402
from models.demos.qwen3_tts.tt.model_config import talker_config_for_hf_id  # noqa: E402
from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS  # noqa: E402

SR = 24000


def decode_icl(ref_codes, codes, decoder_weights):
    """HF generate_voice_clone decode: decode cat([ref, gen]) and cut the reference's share.

    Kept here instead of calling the tree's helper so any commit can be evaluated the
    same way (decode_icl_audio does not exist in every tree).
    """
    full = torch.cat([ref_codes.to(codes.dtype), codes], dim=0)
    audio = S.decode_audio(full, decoder_weights)
    cut = int(ref_codes.shape[0] / full.shape[0] * audio.shape[-1])
    return audio[..., cut:]


data = json.loads(Path(a.sentences).read_text(encoding="utf-8"))
sent = next(s for s in data["sentences"] if s["id"] == a.id)
max_new = sent.get("max_new_tokens", a.max_new_tokens)
ref_wav = HERE / data["reference"]["wav"]
ref_text = (HERE / data["reference"]["text_file"]).read_text(encoding="utf-8").strip()
tag = f"{a.id}_seed{a.seed}"
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

print(f"RUN {tag}: {sent['text']}", flush=True)
main_weights, decoder_weights = S.load_weights(a.hf_id)
tokenizer = AutoTokenizer.from_pretrained(a.hf_id, trust_remote_code=True)

device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=200000000, num_command_queues=1
)
device.enable_program_cache()
try:
    model = Qwen3TTS(device=device, state_dict=main_weights, talker_config=talker_config_for_hf_id(a.hf_id))
    cfg = S.TTSConfig()
    cfg.hidden_size = model.talker_config.hidden_size
    cfg.max_new_tokens = max_new
    # Must exist next to the wav: encode_reference_audio otherwise shells out to ffmpeg.
    ref_codes, ref_audio = S.encode_reference_audio(str(ref_wav))
    spk = model.extract_speaker_embedding(ref_audio)
    emb, trail, pad, cpe = S.create_icl_embedding_ttnn(
        target_text=sent["text"],
        ref_text=ref_text,
        ref_codes=ref_codes,
        speaker_embedding=spk,
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=cfg,
        main_weights=main_weights,
        language="japanese",
    )
    torch.manual_seed(a.seed)
    t0 = time.time()
    codes, _ = S.generate_codes_ttnn(
        model=model,
        device=device,
        inputs_embeds_tt=emb,
        trailing_text_hidden=trail,
        tts_pad_embed=pad,
        code_pred_embeds=cpe,
        config=cfg,
    )
    gen_s = time.time() - t0
    assert codes is not None, "generate_codes_ttnn returned no codes"
finally:
    ttnn.close_mesh_device(device)

wav = decode_icl(ref_codes, codes, decoder_weights).squeeze().float().numpy()
sf.write(out / f"{tag}.wav", wav, SR)
torch.save(codes, out / f"{tag}.codes.pt")
frames = int(codes.shape[0])
res = dict(
    tag=tag,
    id=a.id,
    seed=a.seed,
    text=sent["text"],
    frames=frames,
    max_new_tokens=max_new,
    hit_max=frames >= max_new,
    # Same tree + seed reproduces bit-identical codes, so a changed hash means the change moved the numerics.
    codes_sha1=hashlib.sha1(codes.to(torch.int64).contiguous().numpy().tobytes()).hexdigest()[:16],
    audio_sec=round(len(wav) / SR, 3),
    gen_s=round(gen_s, 2),
)
# Written last: its presence is what marks this generation as done.
(out / f"res_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"OK {tag} frames={frames} audio={res['audio_sec']}s gen={res['gen_s']}s", flush=True)
