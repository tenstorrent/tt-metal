# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A diverse, multilingual request corpus, packed into fixed-length prefill sequences with any tokenizer.

``download`` writes one JSONL per public source to <workdir>/raw/<source>.jsonl (records are chat turns or
documents, tagged with domain, language and source). It is model-independent: download once, reuse for
every model. ``pack`` tokenizes with a model's tokenizer and chat format and writes, per source,
<workdir>/sequences/<name>/{metadata.json, token_meta.npz, samples.json}:
  metadata.json   token_ids (exactly ``tokens``), the layout the prefill harness reads as PREFILL_TRACE_DIR
  token_meta.npz  sample_of[t] = index of the request token t belongs to
  samples.json    one entry per request: source, domain, lang, kind, tokens

Every request starts like a fresh request: chats are rendered with the chat template, documents get the
template's conversation-start prefix. Sequences: one per source (capped per request so one long document
cannot dominate), two interleaved mixes, three synthetic edge cases, and optionally a golden control trace.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import string
import sys
import traceback
from pathlib import Path

import numpy as np

DOMAINS = ("chat", "code", "knowledge", "math", "science", "writing", "multilingual")
WIKI_LANGS = ("en", "zh", "de", "es", "fr", "ja", "ru", "ar", "hr", "ko", "hi")
SEED = 20261110


# --- download ----------------------------------------------------------------------------------
def _sources(load_dataset, rng):
    def chat(user, assistant=None):
        m = [{"role": "user", "content": user.strip()}]
        if assistant:
            m.append({"role": "assistant", "content": assistant.strip()})
        return dict(kind="chat", messages=m)

    def doc(text):
        return dict(kind="doc", text=text.strip())

    def stream(name, config=None, split="train", **kw):
        return load_dataset(name, config, split=split, streaming=True, **kw)

    def shuffled(ds, buffer=2000):
        return ds.shuffle(seed=SEED, buffer_size=buffer)

    def gsm8k():
        for r in shuffled(stream("openai/gsm8k", "main", "train")):
            yield chat(r["question"], r["answer"])

    def competition_math():
        subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        its = [iter(shuffled(stream("EleutherAI/hendrycks_math", s, "train"), 500)) for s in subjects]
        while its:
            for it in list(its):
                try:
                    r = next(it)
                    yield chat(r["problem"], r["solution"])
                except StopIteration:
                    its.remove(it)

    def mgsm():
        langs = ["bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
        rows = {}
        for lang in langs:
            try:
                rows[lang] = list(load_dataset("juletxara/mgsm", lang, split="test"))
            except Exception as e:  # one missing language must not lose the source
                print(f"mgsm {lang}: {e}", file=sys.stderr)
        for i in range(250):
            for lang, rs in rows.items():
                if i < len(rs):
                    r = rs[i]
                    yield dict(chat(r["question"], r.get("answer") or str(r.get("answer_number"))), lang=lang)

    def humaneval_mbpp():
        he = list(load_dataset("openai/openai_humaneval", split="test"))
        mb = list(load_dataset("google-research-datasets/mbpp", "full", split="train"))
        for i in range(max(len(he), len(mb))):
            if i < len(he):
                yield chat(
                    "Complete the following Python function.\n\n" + he[i]["prompt"],
                    he[i]["prompt"] + he[i]["canonical_solution"],
                )
            if i < len(mb):
                yield chat(
                    mb[i]["text"] + "\nYour code should pass these tests:\n" + "\n".join(mb[i]["test_list"]),
                    mb[i]["code"],
                )

    def python_files():
        for r in shuffled(stream("codeparrot/codeparrot-clean-valid", split="train")):
            if 2_000 < len(r["content"]) < 40_000:
                yield doc(r["content"])

    def writing():
        for r in shuffled(stream("euclaise/writingprompts", split="train")):
            if len(r["story"]) > 1500:
                yield chat(r["prompt"].replace("[ WP ]", "").strip(), r["story"].replace("<newline>", "\n"))

    def science():
        sciq = iter(shuffled(stream("allenai/sciq", split="train")))
        arc = iter(shuffled(stream("allenai/ai2_arc", "ARC-Challenge", split="train"), 500))
        pub = iter(shuffled(stream("qiaojin/PubMedQA", "pqa_labeled", split="train"), 500))
        while True:
            r = next(sciq)
            if r["support"]:
                yield chat(f"{r['support']}\n\nQuestion: {r['question']}", r["correct_answer"])
            r = next(arc)
            opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
            yield chat(f"{r['question']}\n{opts}", r["answerKey"])
            r = next(pub)
            yield chat(
                "Context:\n" + "\n".join(r["context"]["contexts"]) + f"\n\nQuestion: {r['question']}", r["long_answer"]
            )

    def mmlu():
        for r in shuffled(stream("cais/mmlu", "all", split="test"), 5000):
            opts = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(r["choices"]))
            yield chat(f"Subject: {r['subject'].replace('_', ' ')}\n{r['question']}\n{opts}", "ABCD"[r["answer"]])

    def trivia():
        for r in shuffled(stream("mandarjoshi/trivia_qa", "rc.nocontext", split="train")):
            yield chat(r["question"], r["answer"]["value"])

    def wiki(lang):
        def gen():
            for r in shuffled(stream("wikimedia/wikipedia", f"20231101.{lang}"), 1000):
                if len(r["text"]) > 3000:
                    yield doc(f"{r['title']}\n\n{r['text']}")

        return gen

    def mmmlu():
        langs = [
            "AR_XY",
            "BN_BD",
            "DE_DE",
            "ES_LA",
            "FR_FR",
            "HI_IN",
            "ID_ID",
            "IT_IT",
            "JA_JP",
            "KO_KR",
            "PT_BR",
            "SW_KE",
            "YO_NG",
            "ZH_CN",
        ]
        its = {l: iter(shuffled(stream("openai/MMMLU", l, split="test"), 2000)) for l in langs}
        while its:
            for l, it in list(its.items()):
                try:
                    r = next(it)
                except StopIteration:
                    del its[l]
                    continue
                opts = "\n".join(f"{k}. {r[k]}" for k in "ABCD")
                yield dict(chat(f"{r['Question']}\n{opts}", r["Answer"]), lang=l.split("_")[0].lower())

    def ultrachat():
        for r in shuffled(stream("HuggingFaceH4/ultrachat_200k", split="test_sft")):
            yield dict(kind="chat", messages=[dict(role=m["role"], content=m["content"]) for m in r["messages"]])

    def oasst():
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        kids, roots = {}, []
        for r in ds:
            if r["parent_id"] is None:
                roots.append(r)
            else:
                kids.setdefault(r["parent_id"], []).append(r)
        rng.shuffle(roots)
        for root in roots:  # first-ranked thread of every root prompt, any language
            msgs, node = [], root
            while node is not None:
                msgs.append(dict(role="user" if node["role"] == "prompter" else "assistant", content=node["text"]))
                ch = sorted(kids.get(node["message_id"], []), key=lambda x: (x["rank"] is None, x["rank"] or 0))
                node = ch[0] if ch else None
            if len(msgs) >= 2:
                yield dict(kind="chat", messages=msgs, lang=root["lang"])

    sources = {
        "math_gsm8k_en": ("math", "en", gsm8k),
        "math_competition_en": ("math", "en", competition_math),
        "math_mgsm_multi": ("math", "multi", mgsm),
        "code_problems_en": ("code", "en", humaneval_mbpp),
        "code_python_files": ("code", "en", python_files),
        "writing_stories_en": ("writing", "en", writing),
        "science_qa_en": ("science", "en", science),
        "knowledge_mmlu_en": ("knowledge", "en", mmlu),
        "knowledge_trivia_en": ("knowledge", "en", trivia),
        "knowledge_mmmlu_multi": ("knowledge", "multi", mmmlu),
        "chat_ultrachat_en": ("chat", "en", ultrachat),
        "chat_oasst_multi": ("chat", "multi", oasst),
    }
    for lang in WIKI_LANGS:
        sources[f"wiki_{lang}"] = ("knowledge" if lang == "en" else "multilingual", lang, wiki(lang))
    return sources


def download(workdir: Path, only=None, max_chars: int = 600_000) -> dict:
    """Public, parquet-backed datasets only (no remote code). Existing sources are kept."""
    from datasets import load_dataset

    raw = workdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    failed, done = [], []
    for name, (domain, lang, fn) in _sources(load_dataset, random.Random(SEED)).items():
        if only and name not in only:
            continue
        path = raw / f"{name}.jsonl"
        if path.exists():
            done.append(name)
            continue
        tmp, chars, n = path.with_suffix(".tmp"), 0, 0
        try:
            with tmp.open("w") as f:
                for rec in fn():
                    rec.setdefault("lang", lang)
                    rec.update(domain=domain, source=name)
                    size = len(rec.get("text", "")) + sum(len(m["content"]) for m in rec.get("messages", []))
                    if size == 0:
                        continue
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    chars += size
                    n += 1
                    if chars >= max_chars:
                        break
            tmp.replace(path)
            done.append(name)
            print(f"ok {name}: {n} records, {chars} chars", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAILED {name}\n{traceback.format_exc()}", flush=True)
    return dict(done=done, failed=failed)


# --- chat formats --------------------------------------------------------------------------------
def glm_template_ids(tok, messages):
    """GLM-family chat format (GLM-5.3 ships no chat_template): [gMASK]<sop>, then <|role|>\\n content per
    turn; past assistant turns carry an empty think block."""
    text = "[gMASK]<sop>"
    for m in messages:
        if m["role"] == "assistant":
            text += "<|assistant|>\n<think></think>\n" + m["content"]
        else:
            text += f"<|{m['role']}|>\n" + m["content"]
    return tok.encode(text, add_special_tokens=False)


class Renderer:
    """Turns raw records into token ids the way a serving stack would."""

    def __init__(self, tokenizer_path: str, chat_format: str = "hf"):
        from transformers import AutoTokenizer

        self.tok = tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=chat_format == "hf")
        self.chat_format = chat_format
        self.special = set(tok.all_special_ids)
        self.vocab_len = max(len(tok), max(self.special) + 1)
        if chat_format == "hf":
            # conversation-start prefix = everything the template emits before the first user message
            # (e.g. Kimi-K3's fixed thinking-effort system message)
            sentinel = "QZJ7SENTINEL7JZQ"
            rendered = tok.apply_chat_template([{"role": "user", "content": sentinel}], tokenize=False)
            head = rendered[: rendered.index(sentinel)]
            marker = max(
                head.rfind('<|open|>message role="user"'), head.rfind("<|im_start|>user"), head.rfind("<|user|>")
            )
            self.prefix = tok.encode(head[: marker if marker > 0 else len(head)], add_special_tokens=False)
        elif chat_format == "glm":
            self.prefix = []
            for t in glm_template_ids(tok, [{"role": "user", "content": "x"}]):
                if t not in self.special or tok.convert_ids_to_tokens(t) in ("<|user|>",):
                    break
                self.prefix.append(t)
        else:
            raise ValueError(f"unknown chat format {chat_format!r} (hf | glm)")

    def chat(self, messages):
        if self.chat_format == "hf":
            return self.tok.encode(self.tok.apply_chat_template(messages, tokenize=False), add_special_tokens=False)
        return glm_template_ids(self.tok, messages)

    def render(self, rec):
        if rec["kind"] == "chat":
            return self.chat(rec["messages"])
        return self.prefix + self.tok.encode(rec["text"], add_special_tokens=False)


# --- synthetic edge cases ------------------------------------------------------------------------
def edge_repetition():
    recs = []
    for w in ["the", "of", "1", "0", " ", "\n", "a", "=", "def", "的"]:
        recs.append(
            dict(
                kind="doc",
                text=(w + (" " if w.strip() else "")) * 2500,
                source="edge_repetition",
                domain="edge",
                lang="none",
            )
        )
    recs.append(
        dict(
            kind="doc",
            text="The quick brown fox jumps over the lazy dog. " * 600,
            source="edge_repetition",
            domain="edge",
            lang="en",
        )
    )
    recs.append(
        dict(
            kind="chat",
            messages=[
                {"role": "user", "content": "Repeat 'hello world' many times."},
                {"role": "assistant", "content": "hello world " * 1500},
            ],
            source="edge_repetition",
            domain="edge",
            lang="en",
        )
    )
    return recs * 4


def edge_structured():
    recs = []
    r = random.Random(7)

    def word():
        return "".join(r.choice(string.ascii_lowercase) for _ in range(r.randint(3, 10)))

    for i in range(400):
        kind = i % 8
        if kind == 0:
            text = json.dumps(
                [
                    {
                        "id": r.randint(1, 10**6),
                        "name": word().title(),
                        "email": f"{word()}@{word()}.com",
                        "score": round(r.random() * 100, 3),
                        "tags": [word() for _ in range(3)],
                        "active": r.random() > 0.5,
                    }
                    for _ in range(25)
                ],
                indent=2,
            )
        elif kind == 1:
            text = "date,region,product,units,revenue\n" + "\n".join(
                f"2026-{r.randint(1,12):02d}-{r.randint(1,28):02d},{word()},{word()},{r.randint(1,999)},{r.random()*1e4:.2f}"
                for _ in range(80)
            )
        elif kind == 2:
            text = base64.b64encode(r.randbytes(1500)).decode()
        elif kind == 3:
            text = "\n".join(f"{0x1000*j:08x}: " + " ".join(f"{b:02x}" for b in r.randbytes(16)) for j in range(60))
        elif kind == 4:
            text = "\n".join(
                f"https://{word()}.{r.choice(['com','org','io','net'])}/{word()}/{word()}?id={r.randint(1,10**9)}&ref={word()}"
                for _ in range(60)
            )
        elif kind == 5:
            emo = "😀😂🥲😍🤔🙃🔥✨🎉🚀🌍🍕⚽🎵💡📈✅❌🇭🇷🇯🇵"
            text = " ".join("".join(r.choice(emo) for _ in range(r.randint(1, 4))) + " " + word() for _ in range(300))
        elif kind == 6:
            text = (
                "| "
                + " | ".join(word() for _ in range(6))
                + " |\n|"
                + "---|" * 6
                + "\n"
                + "\n".join("| " + " | ".join(str(r.randint(0, 9999)) for _ in range(6)) + " |" for _ in range(60))
            )
        else:
            text = "\n".join(
                f"$$\\int_{{{r.randint(0,9)}}}^{{{r.randint(10,99)}}} \\frac{{x^{{{r.randint(2,9)}}} + {r.randint(1,9)}}}{{\\sqrt{{x+{r.randint(1,9)}}}}}\\,dx = \\sum_{{k=0}}^{{\\infty}} \\binom{{n}}{{k}} \\alpha_k$$"
                for _ in range(30)
            )
        recs.append(
            dict(
                kind="chat",
                messages=[{"role": "user", "content": "Parse and summarize this data:\n\n" + text}],
                source="edge_structured",
                domain="edge",
                lang="none",
            )
        )
    return recs


# --- pack ----------------------------------------------------------------------------------------
def _shuffled(xs, seed):
    xs = list(xs)
    random.Random(seed).shuffle(xs)
    return xs


def _interleave(*lists):
    its = [iter(l) for l in lists]
    while its:
        for it in list(its):
            try:
                yield next(it)
            except StopIteration:
                its.remove(it)


def write_sequence(workdir: Path, name: str, ids, sample_of, samples, vocab_len: int | None = None) -> dict:
    n = len(ids)
    if vocab_len is not None and not (0 <= min(ids) and max(ids) < vocab_len):
        raise ValueError(f"{name}: token ids outside the vocabulary")
    d = workdir / "sequences" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps(dict(token_ids=list(map(int, ids)), n_prompt_tokens=n, run=name)))
    np.savez_compressed(d / "token_meta.npz", sample_of=np.asarray(sample_of, dtype=np.int32))
    (d / "samples.json").write_text(json.dumps(samples, ensure_ascii=False))
    langs = {}
    for s in samples:
        langs[s["lang"]] = langs.get(s["lang"], 0) + s["tokens"]
    return dict(samples=len(samples), tokens_sha256=tokens_sha256(ids), lang_tokens=langs, unique_tokens=len(set(ids)))


def tokens_sha256(ids) -> str:
    return hashlib.sha256(json.dumps(list(map(int, ids)), separators=(",", ":")).encode()).hexdigest()


def pack(
    workdir: Path,
    tokenizer: str,
    tokens: int,
    chat_format: str = "hf",
    control_trace: Path | None = None,
    control_name: str = "control",
    cpp_root: Path | None = None,
) -> dict:
    """Pack every downloaded source (plus mixes, edge cases, optional control trace) into ``tokens``-long
    sequences. Returns the index written to sequences/index.json."""
    rd = Renderer(tokenizer, chat_format)
    raw = workdir / "raw"

    def read(name):
        with (raw / f"{name}.jsonl").open() as f:
            return [json.loads(l) for l in f]

    def pack_records(records, cap_per_sample=None):
        ids, sample_of, samples = [], [], []
        for rec in records:
            r = rd.render(rec)
            if cap_per_sample:
                r = r[:cap_per_sample]
            take = r[: tokens - len(ids)]
            sample_of += [len(samples)] * len(take)
            samples.append(
                dict(
                    source=rec.get("source"),
                    domain=rec.get("domain"),
                    lang=rec.get("lang"),
                    kind=rec["kind"],
                    tokens=len(take),
                )
            )
            ids += take
            if len(ids) >= tokens:
                return ids, sample_of, samples
        raise ValueError(f"only {len(ids)} tokens available, need {tokens}")

    def cpp_files():
        files = sorted(
            p
            for p in cpp_root.rglob("*")
            if p.suffix in (".cpp", ".hpp", ".h", ".cc")
            and "third_party" not in p.parts
            and 3000 < p.stat().st_size < 60000
        )
        for p in _shuffled(files, 11):
            yield dict(
                kind="doc",
                text=f"// {p.relative_to(cpp_root)}\n" + p.read_text(errors="replace"),
                source="code_cpp_files",
                domain="code",
                lang="cpp",
            )

    available = sorted(p.stem for p in raw.glob("*.jsonl"))
    runs = {name: (lambda name=name: pack_records(read(name), cap_per_sample=8192)) for name in available}
    if cpp_root:
        runs["code_cpp_files"] = lambda: pack_records(cpp_files(), cap_per_sample=8192)
    runs["edge_repetition"] = lambda: pack_records(edge_repetition())
    runs["edge_structured"] = lambda: pack_records(edge_structured())
    for seed in (1, 2):  # realistic mixed traffic: every source interleaved request by request
        runs[f"mixed_all_s{seed}"] = lambda seed=seed: pack_records(
            _interleave(*[_shuffled(read(n), seed * 1000 + i) for i, n in enumerate(_shuffled(available, seed))]),
            cap_per_sample=4096,
        )
    index = {}
    for name, fn in runs.items():
        try:
            index[name] = write_sequence(workdir, name, *fn(), vocab_len=rd.vocab_len)
            print(json.dumps({name: index[name]}, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps(dict(run=name, error=str(e))), flush=True)
    normal = np.array([t for t in range(rd.tok.vocab_size) if t not in rd.special])
    ids = np.random.default_rng(13).choice(normal, tokens).tolist()
    index["edge_random_tokens"] = write_sequence(
        workdir,
        "edge_random_tokens",
        ids,
        [0] * tokens,
        [dict(source="edge_random_tokens", domain="edge", lang="none", kind="tokens", tokens=tokens)],
    )
    if control_trace:
        ids = json.loads((Path(control_trace) / "metadata.json").read_text())["token_ids"][:tokens]
        if len(ids) < tokens:
            raise ValueError(f"control trace has {len(ids)} tokens, need {tokens}")
        index[control_name] = write_sequence(
            workdir,
            control_name,
            ids,
            [0] * tokens,
            [dict(source=control_name, domain="control", lang="en", kind="trace", tokens=tokens)],
        )
    (workdir / "sequences" / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return index
