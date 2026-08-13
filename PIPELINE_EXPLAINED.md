# How this pipeline works, explained simply

A plain-language walk through what PR #46283's tool does, step by step, using real output from our
Voxtral run as the examples. No jargon without an explanation.

---

## The problem it is trying to solve

You have a model that runs on a normal computer (in PyTorch). You want it to run on a Tenstorrent
chip. Those are different worlds: the chip has its own set of operations, its own memory rules, and
its own way of splitting work across cores.

Normally a person does this by hand:

1. read the model and understand every layer,
2. rewrite each layer using the chip's operations,
3. check the rewritten version produces the same numbers as the original,
4. then spend weeks making it fast.

That is what you did for Voxtral, across 74 recorded experiments.

**This tool tries to do all four steps automatically.** An AI writes the code; a fixed program
checks the work. The AI is never allowed to decide whether it succeeded — that is the whole design.

---

## The one idea you need first: PCC

**PCC** is a single number that answers *"do these two lists of numbers agree?"*

- `1.0` = identical
- `0.99` = very close
- `0.5` = not really related

The tool runs the original PyTorch layer and the new chip layer on the **same input**, then compares
the two outputs. If PCC is at least **0.99**, the new version is accepted.

Real numbers from our run:

```
attention          PCC 0.9998358
decoder_layer      PCC 0.9999971
m_l_p              PCC 0.9999944
r_m_s_norm         PCC 0.9999874
rotary_embedding   PCC 0.9999992      pass mark 0.99
```

---

## The two halves

| half | command | job |
|---|---|---|
| **1. Make it work** | `auto-up` | produce a chip version whose numbers match |
| **2. Make it fast** | `optimize` | make that version quicker, without breaking the numbers |

They run one after the other. Half 1 has six steps.

---

# HALF ONE: making it work

## Step 0 — Can I even open this model?

It checks the model loads at all.

```
transformers can load `/localdev/.../voxtral-tts-backbone` locally   [ok]
```

If this fails, everything else is pointless, so it stops here.

It also makes a **private copy of the whole code repository** in a temporary folder and works there,
so nothing it does can damage your real files.

## Step 1 — What is this model, and does the chip already know how to run it?

Two questions.

**(a) How big is it, and will it fit?** It counts the weights and compares against the chip's memory:

```
3,429,020,008 parameters · 6.86 GB of weights
P150: needs 5.12 GB/chip, 24.08 GB headroom -> FITS (comfortable)
```

**(b) Does Tenstorrent's code library already contain the pieces this model needs?**

A model is built from standard ingredients — attention, normalisation, position encoding, and so on.
The tool lists them and looks for an existing chip implementation of each:

```
STATUS  BLOCK                  EFFORT     TT IMPLEMENTATION
[ ok ]  Token embedding        drop-in    models/tt_transformers/tt/embedding.py
[ ok ]  GQA attention          drop-in    models/tt_transformers/tt/attention.py
[ ok ]  Standard RoPE          drop-in    models/tt_transformers/tt/rope.py
[ ok ]  SwiGLU MLP             drop-in    models/tt_transformers/tt/mlp.py
...
Summary: 11 ready / 0 partial / 0 missing
```

Three possible answers per ingredient:

- **ready** — the library already has it, use it as-is
- **partial** — it exists somewhere, but buried inside another model's folder; you would have to copy
  and adapt that code
- **missing** — nobody has ever written this for this hardware; someone must write new low-level code

**Important:** "missing" means *missing from Tenstorrent's library*, not missing from your model.

Our model scored 11 ready / 0 missing, because it is a standard transformer and all those pieces
were written long ago for Llama, Mistral and Qwen.

## Step 2 — Build an empty project folder

It finds the most similar model already running on the chip and copies its folder layout as a
starting skeleton: a `demo/` folder, a `tt/` folder for chip code, a `tests/` folder.

Nothing works yet. It is an empty house with the rooms marked out.

## Step 3 — Is this a family I recognise?

The tool keeps a list of model families it knows (Llama, Qwen, Whisper, BERT, and so on). If yours
is new, an AI writes a short description of how to handle this family and adds it to the list, so
the *next* model of the same kind is easier.

Our model matched the Mistral family:

```
Backend match: EXACT (Voxtral TTS Backbone (mistral decoder)) via model_type='mistral'
```

## Step 4 — Fill every room with a placeholder

Before writing any chip code, it fills every component with a **stand-in that runs on the normal
computer**. The model works end to end immediately — just slowly, because nothing is on the chip
yet.

This matters: there is always something runnable, so progress can be measured as
"how much has moved onto the chip".

## Step 5 — Record what the right answer looks like

**This is the most important step and easy to miss.**

It runs the original PyTorch model once and **writes down the real inputs and outputs of every
single layer**.

```
[preflight] running HF model once to capture REAL IO tensors for 5 pending component(s)
[preflight] captured 5/5 components; per-component PCC tests will use real inputs
```

Why it matters: to test whether your new attention layer is correct, you feed it something and
compare against the original. If you feed both *random numbers*, you get a misleading answer,
because random numbers do not look like the real thing. Your own project learned this the hard way —
random inputs scored 0.892 where real inputs scored 0.9994, and it sent someone chasing a problem
that did not exist.

So: real inputs, captured once, used for every test afterwards.

## Step 6 — The loop that does the actual work

Now it takes the model apart into **components** and works on them one at a time.

Ours split into five:

```
decoder_layer      NEW     <- must be written from scratch
attention          REUSE   <- point at the library's existing version
m_l_p              REUSE
r_m_s_norm         REUSE
rotary_embedding   REUSE
```

Three kinds of component:

- **REUSE** — the library already has this exact thing; just wire it up
- **ADAPT** — copy a similar model's version and modify it
- **NEW** — nothing suitable exists; the AI writes fresh chip code

*(Note: a component can be NEW even when all the ingredients were "ready" in step 1. Step 1 asks
"does the library have attention at all?"; step 6 asks "does this particular model's layer need its
own code written?")*

### What one round of the loop looks like

```
1. ask "what should I work on?"        -> "work on decoder_layer"
2. AI writes chip code for it
3. run its test on the actual chip     -> PCC 0.9999971
4. is PCC >= 0.99?
      yes -> mark it finished, save a copy of the working code
      no  -> throw the change away, put it back in the queue, try again
5. repeat until every component is finished
```

If a component keeps failing, the tool escalates to a stronger (more expensive) AI model, and if it
still fails, either splits it into smaller pieces or leaves it running on the normal computer so the
rest of the model still works.

The saved copy is called a **graduation snapshot** — proof this version worked, so a later mistake
can always be rolled back.

### How the AI is kept honest

The AI is given a set of **tools** — small commands it can call:

```
termination_check    what should I work on next? am I done?
run_component        run this component's test on the chip
record_result        write down that it passed or failed
fall_back_to_cpu     give up on this one, leave it on the normal computer
```

The AI chooses *what to try*. The tools decide *whether it worked*. The AI cannot mark its own
homework.

*(This is also what broke in our run: the program providing those tools failed to start because a
required package was missing, so the AI had no way to ask questions or report results. It did the
work anyway using ordinary file editing, but the system never found out. See `TOOL_FINDINGS.md`, F6.)*

## After step 6 — wire it together and grade it (`emit-e2e`)

Finished components are separate pieces. This stage connects them into a working whole, then a
**second, independent AI** re-runs everything from scratch and tries to find holes. Three checks:

1. does it run without secretly falling back to the normal computer?
2. is every finished component actually being used?
3. do the final numbers still match?

If the grader says no, a third AI fixes the gaps and the grader runs again. The point of a separate
grader is simple: the AI that wrote the code is a bad judge of it.

---

# HALF TWO: making it fast (`optimize`)

Now the model is correct but slow. This half is a loop too.

## Step 1 — Measure where the time goes

It runs the model on the chip with a profiler and gets, for every operation: how long it took, and
**what it was waiting for**:

- **waiting for maths** — the calculating units are the limit
- **waiting for memory** — reading the weights from memory is the limit
- **waiting for the computer** — the chip finished and is idle while the host prepares the next
  instruction

That last one matters more than people expect.

## Step 2 — Pick the worst offender

It groups operations into families (matrix multiplies, data movement, and so on), works out the
theoretical fastest possible time, and attacks whichever group has the most time available to win.

## Step 3 — Climb a fixed ladder of tricks

For the chosen operation it tries fixes **in a set order**, cheapest first:

```
grid       use all the chip's cores instead of some
dtype      store the weights in fewer bits (smaller = faster to read, slightly less accurate)
shard      split a piece of data across cores' local memory instead of main memory
fidelity   let the maths be less precise (faster, slightly less accurate)
host       reduce the waiting-for-the-computer overhead
fusion     merge operations so intermediate results never make a round trip to memory
tt-lang    hand-write a custom operation in their kernel language
C++        hand-write a raw low-level operation. Last resort.
```

The order changes depending on what the operation was waiting for. If it is waiting on memory, try
the tricks that move fewer bytes first. If it is waiting on the computer, none of those help — go
straight to the overhead fixes.

## Step 4 — Three checks before keeping anything

After each change:

1. **is it still correct?** (PCC)
2. **is it actually faster?**
3. **is the measurement trustworthy?**

That third one is subtle. If a change crashes the model halfway, the profiler happily reports a
wonderful time — because half the work never ran. The tool detects that the operation count
dropped and rejects the result. Its own instructions say:

> *"You may want a change to be a win; the tools decide, not you."*

Pass all three, keep it. Fail any, undo it. Either way, record that the trick was tried.

## Step 5 — Measure again and repeat

After a win, the bottleneck moves somewhere else. So it re-measures and goes back to step 2. It
stops when every operation is at its theoretical floor or every trick has been tried — not after a
fixed number of attempts.

## It remembers

A trick the AI invents that works gets written into a catalogue as "unproven". If it later works on
a **different** model, it is promoted to "trusted". Next time, the AI looks up the catalogue before
inventing anything, so the tool gets faster at getting models fast.

---

# The whole thing in one picture

```
  your model (PyTorch)
        |
   [0] can I open it?
   [1] how big is it? does the library have the pieces?
   [2] make an empty project folder
   [3] recognise the family (or learn a new one)
   [4] fill it with normal-computer placeholders  -> runs, but slow
   [5] record the right answers from the original
        |
   [6] LOOP: pick a component -> AI writes chip code -> test on chip
              PCC >= 0.99 ? keep it : throw it away and retry
        |
        v  all components on the chip
   emit-e2e: wire them together, a second AI grades it, a third fixes holes
        |
        v  works end to end
   optimize: LOOP: measure -> find slowest -> try next trick on the ladder
                    correct AND faster AND honest ? keep : undo
        |
        v
   runs on the chip, correctly and fast
```

---

# Glossary

| word | what it means |
|---|---|
| **PCC** | a score from 0 to 1 for "do these two sets of numbers agree". 0.99 is the pass mark |
| **component** | one piece of the model, e.g. the attention layer |
| **block** | one *capability* the model needs, e.g. "attention". Checked against the library in step 1 |
| **graduate** | a component passed its test and its code was saved |
| **snapshot** | the saved copy of code that worked, so you can roll back |
| **stub** | the file holding one component's chip code |
| **fall back to CPU** | give up on a component; leave it on the normal computer so the rest works |
| **tool (MCP tool)** | a command the AI can call, like "run this test". The AI asks, the program does it |
| **worktree** | a private copy of the code repository, so experiments cannot damage the real one |
| **kernel** | a small program that runs directly on the chip's cores |
| **roofline** | the theoretically fastest this operation could ever be, given the hardware |
| **dtype** | how many bits each number uses. Fewer bits = faster to read, less precise |
| **sharding** | splitting one piece of data across many cores |
| **fusion** | merging operations so intermediate results stay on the chip |
