# Blinded qualitative screen

Scored all 12 anonymized outputs using the supplied fixed rubric. Only seed 0 is represented for each of six paired prompts. This is a screening result, not a statistical comparison, stage pass, or backend attribution. No label/key files were accessed.

Ten outputs are coherent but cap-limited; two French translations reach a correct final answer. Expected thinking delimiters, task-relevant quotations, story-opening reuse, and cap-truncated code are not flagged.

## Potential failure requiring control

Sample `a3f42bf74ae77ced2575`, **prompt04**: “What are the three laws of thermodynamics?” contains “Entropy of an isolated system always increases over time (tends toward maximum disorder).” The general law is that entropy never decreases; it can stay constant. Flag: `materially_wrong_or_irrelevant_answer`, treated as **potential**, because this statement occurs in unfinished reasoning. Missing the third law and final answer is separately cap-limited incompleteness.

Before debugging or attributing this observation, run the mandatory **at least 10 matched seeds per prompt on each side**, with identical model-card sampling settings, template, and token limit, then blind-score under the fixed rule. The predeclared comparative threshold is a failure-rate gap greater than 20 percentage points. A single sample cannot establish that gap.

## Per-sample record

| Sample ID | Prompt | Flags | Cap-limited | Evidence and assessment |
|---|---|---|---|---|
| `a3f42bf74ae77ced2575` | prompt04 | materially_wrong_or_irrelevant_answer | Yes | “The change in internal energy of a system equals the heat added to the system minus the work done by the system.” — Potential factual failure: "always increases" excludes constant entropy in equilibrium or an ideal reversible process; the correct general rule is never decreases. This is an uncorrected statement in unfinished reasoning, not a completed final answer. First-law content is sound. Missing third law and final answer are recorded only as cap-limited incompleteness. The flag requires matched-seed control before any attribution or debugging. |
| `d2077874c0959a81ea50` | prompt05 | None | No | “Bonjour, comment allez-vous aujourd'hui ?” — Correct polite French translation. Informal alternative in reasoning is also appropriate. Quoting the source sentence is task-relevant, and the think delimiter is expected. |
| `dc1d4a003e52a6e7cc8a` | prompt02 | None | Yes | “Supervised: spam detection (labeled data saying which emails are spam)” — Relevant, coherent examples distinguish labeled training from grouping without labels. Stops mid-reasoning before a final explanation; this alone is not failure. |
| `bb92fdf0f482c8d7386f` | prompt01 | None | Yes | “teach-ing (2) finds (1) pat-terns (2) = 5” — Coherent haiku drafting. An initial incorrect count for patterns is followed by explicit recounting and a valid five-syllable candidate. Stops while drafting the third line. No finished haiku can be judged; draft revision is not mechanical repetition. |
| `05d8da18efae21a71b4b` | prompt05 | None | No | “Bonjour, comment allez-vous aujourd'hui ?” — Correct formal translation and useful informal alternative in final output. The reasoning loosely describes comment ça va as using tu although that phrase contains no tu; this does not materially affect the correct final answer. Expected French and think delimiter are not failure flags. |
| `ec07faf73273d25d88de` | prompt03 | None | Yes | “a curious young inventor who discovered a jar filled with tiny,” — Relevant story planning followed by a coherent opening that stops mid-sentence. Reusing the supplied story opening is appropriate continuation framing, not prompt-echo failure. No complete narrative under this cap. |
| `9b97a93e9e72d3ee4f13` | prompt02 | None | Yes | “You give it input AND the correct output/answer” — Accurate, accessible contrast and examples in reasoning. No final answer before the cap; the useful reasoning remains coherent. |
| `99a50fcfb88db5e0f66e` | prompt03 | None | Yes | “that they were allergic to silence.” — Imaginative, relevant fairy-tale continuation. A few whimsical phrases are odd but not incoherent or gibberish. Ends mid-word/phrase at the token cap; not evidence of corrupt subwords. No completed story arc. |
| `a7e3646c645c46e1bb3c` | prompt06 | None | Yes | “def fibonacci(n):” — Relevant programming plan and start of an iterative implementation. Code is truncated before initialization finishes and cannot be judged as a complete runnable function. A planned sequence-returning option has not appeared. These are cap limitations, not a proven algorithmic failure. |
| `9d410bf606234d60a6bc` | prompt06 | None | Yes | “The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...” — Correct sequence and coherent discussion of iterative, recursive, memoized, and generator approaches. No function is emitted before the cap; not a failure by this rubric. |
| `b03ef41a54f2b2e6e0c6` | prompt01 | None | Yes | “Pat-tens (2) + in (1) + the (1) + da-ta (2) = 6.” — Provisional haiku counts contain errors: Patterns in the data and Weighing what was before each normally have six syllables, initially marked five. The text explicitly catches the first error and continues revising. No final poem is emitted, so draft mistakes are noted without treating this as a materially wrong completed answer. No mechanical loop is established. |
| `f498904436b228c31335` | prompt04 | None | Yes | “Entropy of an isolated system never decreases” — Correct zeroth- and first-law descriptions and appropriate nondecreasing-entropy second-law statement. Stops before third law/final answer. This is coherent incomplete reasoning, not failure. |

## Within-prompt observations

- **prompt01:** Both samples remain in haiku drafting. `bb92fdf0f482c8d7386f` develops two candidate lines; `b03ef41a54f2b2e6e0c6` proposes three lines then revisits faulty counts. Interim count errors are recorded, with no completed poem to grade.
- **prompt02:** `dc1d4a003e52a6e7cc8a` begins with response structure and analogies; `9b97a93e9e72d3ee4f13` moves directly into labeled versus unlabeled training. Both remain coherent reasoning at cap.
- **prompt03:** `ec07faf73273d25d88de` spends more of its output planning before the discovery of a jar; `99a50fcfb88db5e0f66e` begins narrative earlier with an allergy to silence. Both stories are incomplete.
- **prompt04:** The second-law wording differs: `a3f42bf74ae77ced2575` says “always increases,” whereas `f498904436b228c31335` says “never decreases.” Both omit the third law before cap. The factual wording merits control; shared incompleteness is not itself a failure.
- **prompt05:** Both reach the same correct formal French translation. `05d8da18efae21a71b4b` adds an informal final alternative; `d2077874c0959a81ea50` gives just the formal final translation.
- **prompt06:** `a7e3646c645c46e1bb3c` starts a function before cap; `9d410bf606234d60a6bc` is still planning. Neither provides a complete function in the available output. No algorithmic defect can be established from the truncated material.

Exact sampled-token agreement is not a criterion. Differences above describe visible content and progression, without inferring hidden labels. The JSON artifact includes every sample, concrete excerpts, separate cap status, and the required follow-up.
