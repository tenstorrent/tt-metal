# Qwen3-TTS Japanese regression check

Checks whether a change to `models/demos/qwen3_tts` degrades Japanese output.
Qwen3-TTS reads a fixed set of Japanese sentences in a fixed reference voice,
Whisper transcribes the audio, and the transcripts are scored against the input
text (CER). Run it on the commit before the change and on the commit after it,
then compare the two runs.

It is a tool for a human reviewer, **not** a pass/fail test: generation is
sampled and the aspects that matter most (pitch accent, prosody, number
readings) cannot be judged automatically. The scripts are deliberately not named
`test_*.py`, so pytest does not collect them.

## Usage

From the repository root:

```bash
# 1. Run on the commit before the change and on the commit after it.
python models/demos/qwen3_tts/tests/jp_quality/run_jp_quality.py run --tree /path/to/before-tree --out /tmp/jpq_before
python models/demos/qwen3_tts/tests/jp_quality/run_jp_quality.py run --tree /path/to/after-tree --out /tmp/jpq_after

# 2. Compare them. Writes /tmp/jpq_after/compare.md
python models/demos/qwen3_tts/tests/jp_quality/run_jp_quality.py compare /tmp/jpq_before /tmp/jpq_after
```

`--tree` is the checkout whose `models/demos/qwen3_tts` is tested. It can be a
PR worktree, or one checkout switched between the two commits with `git checkout`.
For example, for a PR branch:

```bash
git worktree add /tmp/before $(git merge-base origin/<base-branch> origin/<pr-branch>)
git worktree add /tmp/after  origin/<pr-branch>
```

The scripts are taken from this checkout, not from `--tree`, so the commits under
test do not need to contain them.

- One run is 14 sentences × 3 seeds = 42 generations, run one after another. Transcription runs on CPU.
- **Resume:** if a generation fails, the run stops; re-running the same command skips finished generations and transcripts. An `--out` directory belongs to one commit, and reusing it for another commit is refused.
- **Individual steps:** `run` is `gen` + `score` + `report`. `score` reuses existing transcripts and only recomputes CER, so after editing `accept` just re-run `score`, then `report` / `compare`.
- **Subset:** narrow a run with e.g. `--only s01_head_hai,s09_num_arabic --seeds 0`.

## Reading the comparison (`compare.md`)

- **Bit-identical count.** The same commit and seed always reproduce bit-identical codes. A sample that is identical before and after is unaffected by the change. If all samples are identical, the change does not alter Japanese generation. Only the samples that differ are listed, with both transcripts and the paths of both wavs for listening.
- **CER.** Character error rate with punctuation and whitespace removed, taken against the closest spelling listed in `accept`. Generation is sampled, so treat a single-sample difference as a prompt to look, not as a verdict.
- **⚠** marks a sentence where, in the after run:
  - a sample was truncated at `max_new_tokens` (✂)
  - a sample contains one of Whisper's stock hallucination phrases (👻)
  - a sample has CER ≥ 0.3
  - a sample's CER rose by 0.1 or more
  - the mean CER rose by more than 0.05
- **Settings must match.** `compare` refuses runs made with a different model or sentence set, since the differences would then not come from the change.

Each run also has its own `report.md` with absolute scores, all transcripts and a
listening checklist of aspects CER cannot judge (pitch accent, question-final
rise, shortened long vowels, number readings).

## Sentence set (`jp_sentences.json`)

| Focus | Sentences |
|---|---|
| Head loss | s01 (shortest), s02 (「お母さん」→「母さん」 is still a valid word) |
| Long vowels / geminates / palatalised syllables | s03, s04, s05 |
| Pitch accent / prosody | s06 (雨 vs 飴), s07 (question) |
| Numbers / units / dates | s08 (kanji numerals), s09 (Arabic digits + units), s10 (date) |
| Loanwords / Latin letters | s11, s12 |
| Long form | s13 (~10 s), s14 (~15 s) |

Only add spelling variants (kanji vs kana, digit formats) to `accept`. Never add a
mis-reading, or the check stops detecting it.

## Reference voice

`jp_reference.wav` is `common_voice_ja_28360668.mp3` from the Common Voice 17.0
Japanese dev set (CC0-1.0), obtained from `fsicoli/common_voice_17_0`.
`jp_reference.txt` is its transcript, which Qwen3-TTS needs alongside the audio
for voice cloning. `make_reference.py` prepares both:

- converts the clip to 24 kHz mono
- normalises RMS to match `demo/jim_reference.wav`
- writes `jp_reference.refcache.pt`

The refcache is needed because `server.encode_reference_audio` shells out to
ffmpeg whenever no cache exists; with the cache, ffmpeg is never called.
`*.pt` is git-ignored, so commit the refcache with `git add -f`.

## Limitation

- **Decoding lives in this directory.** Like HF `generate_voice_clone`, it decodes `cat([ref, gen])` and cuts off the reference's share of the waveform. It does not use the tree's `decode_icl_audio` or `trim_codec_frames`, so both commits are decoded the same way. As a consequence, changes that only touch the tree's decoding or post-processing are not covered.
