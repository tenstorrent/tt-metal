# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write the id ranking QWEN36_DRAFT_VOCAB_TOPK keeps its top K drafter ids from (tt/draft_vocab.py).

Token counts come in groups, and the ranking is by a weighted sum of each group's normalized frequency:

    score(id) = sum over groups g of  w_g * count_g(id) / tokens_g

so a group's weight is its share of the traffic the kept set is meant for, whatever its corpus size. Groups:
  qwen      --traces: a JSONL of the model's own generations, one object per line with "text" (prompt + completion),
            "prompt" (its prefix) and optionally "finish_reason"; completion and prompt tokens both count (<|im_end|>
            appended to a completion with finish_reason == "stop", as the model emitted it).
  <lang>    --chat: the assistant turns of public multilingual chat data, per language (2-letter code): 7 shards of
            allenai/WildChat-4.8M (real user <-> chatbot traffic, turn-level language labels; labels outside the mapped
            list are detector artifacts such as JSON dumps and are dropped), CohereLabs/aya_dataset (train + test) and
            OpenAssistant/oasst2 (train + validation), fetched with huggingface_hub.
Every text is tokenized with the model's tokenizer, no special tokens added.

Output (np.savez_compressed): rank (uint32, the ids with a nonzero score by score descending then id ascending, at
most --rank-len), always (uint32, the tokenizer's added tokens: always kept), meta (JSON string: sources, weights,
tokens per group). tt/draft_vocab.py keeps ``always``, then ``rank`` in order, then the remaining ids in ascending id.

The shipped tt/draft_vocab_data/draft_vocab_rank_v2.npz:

    python models/demos/blackhole/qwen36/scripts/draft_vocab_counts.py --tokenizer $HF_MODEL --out rank.npz \\
        --traces traces.jsonl --chat --groups qwen=1,en=1,es=0.25,pt=0.25,fr=0.25,de=0.25,it=0.25,ru=0.25

with traces.jsonl the 3000 greedy Qwen3.8-27B completions of public prompts behind v1 (HuggingFaceH4/ultrachat_200k
1200, openai/gsm8k 600, sahil2801/CodeAlpaca-20k 600, allenai/ai2_arc 300, glaiveai/glaive-function-calling-v2 300).
For a deployment's own traffic: --traces <its generations> --groups qwen=1 ranks by its counts alone.
"""

import argparse
import collections
import json

import numpy as np

WILDCHAT = ("allenai/WildChat-4.8M", [f"data/train-{s:05d}-of-00086.parquet" for s in (0, 14, 28, 42, 56, 70, 85)])
AYA = ("CohereLabs/aya_dataset", ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"])
OASST2 = (
    "OpenAssistant/oasst2",
    [
        "data/train-00000-of-00001-88ba0162028a73fc.parquet",
        "data/validation-00000-of-00001-1deeef95c3248fe0.parquet",
    ],
)
WC_CONV_CAP = 40000  # WildChat conversations per conversation-level language (caps English)
MAX_CHARS = 30000  # characters per text (pasted documents)
# WildChat (lingua) language names -> 2-letter codes; other labels are dropped
WC_LANG = {
    "English": "en", "Chinese": "zh", "Russian": "ru", "French": "fr", "Spanish": "es", "Portuguese": "pt",
    "German": "de", "Arabic": "ar", "Italian": "it", "Turkish": "tr", "Japanese": "ja", "Korean": "ko",
    "Vietnamese": "vi", "Polish": "pl", "Indonesian": "id", "Persian": "fa", "Ukrainian": "uk", "Dutch": "nl",
    "Hindi": "hi", "Thai": "th", "Hebrew": "he", "Greek": "el", "Czech": "cs", "Swedish": "sv", "Romanian": "ro",
    "Hungarian": "hu", "Bengali": "bn", "Malay": "ms", "Tagalog": "tl", "Urdu": "ur", "Finnish": "fi",
    "Danish": "da", "Bokmal": "no", "Nynorsk": "no", "Kazakh": "kk", "Uzbek": "uz",
}  # fmt: skip
# Aya ISO 639-3 codes -> 2-letter codes (the languages a group may name; others keep their 3-letter code)
AYA_LANG = {
    "eng": "en", "zho": "zh", "rus": "ru", "fra": "fr", "spa": "es", "por": "pt", "deu": "de", "arb": "ar",
    "ary": "ar", "arz": "ar", "ars": "ar", "apc": "ar", "ajp": "ar", "acm": "ar", "ita": "it", "tur": "tr",
    "jpn": "ja", "kor": "ko", "vie": "vi", "pol": "pl", "ind": "id", "pes": "fa", "ukr": "uk", "nld": "nl",
    "hin": "hi", "tha": "th", "heb": "he", "ell": "el", "ces": "cs", "swe": "sv", "ron": "ro", "hun": "hu",
    "ben": "bn", "zsm": "ms", "tgl": "tl", "fil": "tl", "urd": "ur", "fin": "fi", "dan": "da", "nob": "no",
}  # fmt: skip


def chat_texts(groups):
    """(language, assistant text) of the public chat sources, for the languages in ``groups`` only."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    conv_per_lang = collections.Counter()
    for f in WILDCHAT[1]:
        pf = pq.ParquetFile(hf_hub_download(WILDCHAT[0], f, repo_type="dataset"))
        for batch in pf.iter_batches(batch_size=2000, columns=["conversation", "language"]):
            for row in batch.to_pylist():
                lang = WC_LANG.get(row.get("language") or "")
                if lang is None or conv_per_lang[lang] >= WC_CONV_CAP:
                    continue
                conv_per_lang[lang] += 1
                for turn in row["conversation"] or []:
                    tl = WC_LANG.get(turn["language"]) if turn.get("language") else lang  # turn label, else the conversation's
                    if turn.get("role") == "assistant" and tl in groups and turn.get("content"):
                        yield tl, turn["content"][:MAX_CHARS]
    for f in AYA[1]:
        df = pq.read_table(hf_hub_download(AYA[0], f, repo_type="dataset"), columns=["targets", "language_code"])
        for tgt, lc in zip(df.column("targets").to_pylist(), df.column("language_code").to_pylist()):
            lang = AYA_LANG.get(lc, lc)
            if lang in groups and tgt:
                yield lang, tgt[:MAX_CHARS]
    for f in OASST2[1]:
        df = pq.read_table(hf_hub_download(OASST2[0], f, repo_type="dataset"), columns=["text", "role", "lang"])
        for txt, role, lc in zip(df.column("text").to_pylist(), df.column("role").to_pylist(), df.column("lang").to_pylist()):
            lang = (lc or "").split("-")[0].lower()
            if role == "assistant" and lang in groups and txt:
                yield lang, txt[:MAX_CHARS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", help="JSONL of the model's generations (group 'qwen')")
    ap.add_argument("--chat", action="store_true", help="count the public multilingual chat sources")
    ap.add_argument("--groups", default="qwen=1", help="comma list name=weight (qwen, or 2-letter language codes)")
    ap.add_argument("--vocab-size", type=int, default=248320)
    ap.add_argument("--rank-len", type=int, default=196608, help="ids kept in the stored ranking")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    groups = {k: float(v) for k, v in (kv.split("=") for kv in a.groups.split(","))}
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rt = tok._tokenizer  # the Rust tokenizer: batched encode
    V = a.vocab_size
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    always = sorted(int(i) for i in tok.added_tokens_decoder.keys() if int(i) < V)
    counts = {g: np.zeros(V, dtype=np.int64) for g in groups}
    texts = collections.Counter()

    def add(g, batch):
        for e in rt.encode_batch(batch, add_special_tokens=False):
            ids = np.asarray(e.ids, dtype=np.int64)
            counts[g] += np.bincount(ids[ids < V], minlength=V)
        texts[g] += len(batch)

    if "qwen" in groups:
        assert a.traces, "group 'qwen' needs --traces"
        comp, prom = [], []
        with open(a.traces) as f:
            for line in f:
                r = json.loads(line)
                assert r["text"].startswith(r["prompt"]), "text must start with its prompt"
                comp.append(r["text"][len(r["prompt"]) :])
                prom.append(r["prompt"])
                if r.get("finish_reason") == "stop":
                    counts["qwen"][eos] += 1
        add("qwen", comp)
        add("qwen", prom)
    langs = [g for g in groups if g != "qwen"]
    if langs:
        assert a.chat, f"language groups {langs} need --chat"
        buf = collections.defaultdict(list)
        for lang, txt in chat_texts(set(langs)):
            buf[lang].append(txt)
            if len(buf[lang]) >= 1024:
                add(lang, buf.pop(lang))
        for lang, b in buf.items():
            add(lang, b)
    score = np.zeros(V, dtype=np.float64)
    tokens = {}
    for g, w in groups.items():
        tokens[g] = int(counts[g].sum())
        assert tokens[g] > 0, f"group {g!r} has no tokens"
        score += w * counts[g] / tokens[g]
    obs = np.nonzero(score > 0)[0]
    rank = obs[np.lexsort((obs, -score[obs]))][: a.rank_len]  # score descending, then id ascending
    meta = dict(
        format="draft_vocab_rank_v2",
        groups=groups,
        tokens=tokens,
        texts=dict(texts),
        tokenizer=a.tokenizer.rstrip("/").split("/")[-1],
        vocab_size=V,
        traces=a.traces,
        chat_sources=[WILDCHAT, AYA, OASST2] if langs else [],
        wildchat_conversation_cap=WC_CONV_CAP,
        scored_ids=int(obs.shape[0]),
    )
    np.savez_compressed(
        a.out,
        rank=rank.astype(np.uint32),
        always=np.array(always, dtype=np.uint32),
        meta=np.array(json.dumps(meta)),
    )
    print(f"wrote {a.out}: rank {rank.shape[0]} of {obs.shape[0]} scored ids; tokens per group {tokens}")


if __name__ == "__main__":
    main()
