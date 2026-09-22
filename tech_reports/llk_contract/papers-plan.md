# Papers on the LLK contract: a plan

**State: two papers ready to write, one open.** Paper 1 and paper 2 below are planned in full.
Paper 3 is unresolved, and §"Paper 3: open" records what was considered and why each was set aside,
so the search does not go round again. Paper 4 is parked: it is where this line of work leads, not a
place to start from.

The line of argument the set shares:

> **One contract, several views of it. Paper 1 writes the contract down as maths and checks it in
> software, after the fact. Paper 2 is the API view: a typed register layer where a class of
> violations cannot be spelled. The view still missing is the documentation one, a real reference
> manual for the part, which is the goal but is years of work rather than a paper.**

**Who writes these.** The candidates are digital-design people. That is a constraint on the plan, not
an afterthought, so each paper below carries a "skills this needs" line. Papers 1 and 2 are close
enough to hardware to make the cut as written: paper 1 is a state machine over a register map, which
is native ground, and paper 2 is a measurement of an API that already exists. It is the constraint
that paper 3 keeps failing, which is why it is still open.

Shared assets already in this repo: [`llk-formal-contract.md`](./llk-formal-contract.md) (the
contract), [`san_contract_gaps.md`](./san_contract_gaps.md) (where the code and the contract
differ, plus the measured numbers), and [`related-work.md`](./related-work.md) (full citations and
links for everything named below).

Each paper below has the same headings, so they are easy to compare: **one sentence**, **the
problem**, **what we do**, **the claim we can be wrong about**, **how we show it**, **skills this
needs**, **where it sits in the literature**, and **what a reviewer will push on**.

---

## Paper 1: the LLK contract, and what a checker can see

**One sentence.** Here is a formal contract for a hardware kernel library that never had one, plus
a checker for it, plus an honest measurement of what the checker cannot see.

**The problem.** The LLK is the layer every compute kernel sits on. Its rules are real but unwritten:
call `init` before `execute`, call `uninit` after some ops and not others, do not reconfigure in the
middle of a live operation, do not read a format field nobody set. Break a rule and you get wrong
numbers, not a crash. The rules live in reviewers' heads.

**What we do.** Write the rules as maths. Two parts:

1. A per-unit state machine (one per execution unit: Unpack, FPU, SFPU, Pack). The states are
   `INITIAL / CONFIGURED / INITIALIZED / EXECUTED / UNINITIALIZED / RECONFIGURED`, and the whole
   contract on call order is one transition table (formal §6).
2. A split of the machine state into `Tracked` and `Scratch` (formal §5). `Tracked` is what a later
   call may rely on. `Scratch` is what every `execute` sets fresh, so leaving junk there is fine.
   Four laws, F1 to F4, say what each entry point may do to each half.

Then a Sanitizer that checks it at run time.

**The claim we can be wrong about.** A kernel obeys the contract exactly when every transition is
`ok` in the §6 table and every `execute` leaves `Tracked` where it found it. The interesting half is
the negative one: we can say precisely which violations no correctly-built checker of this shape can
ever catch, and how many there are.

**How we show it.**

- Two measurements of the same thing from opposite directions. A static scan of 435 real kernels
  (3014 state-changing calls) gives a blind-spot rate of at most 1.5%, i.e. 45 `uninit` sites where
  the tool is quiet but correctness rides on something it does not hold (gaps §10.2). An independent
  fuzzer over all call sequences of length 6 (584 scenarios, labelled by an AI auditor that was
  given only §1 to §9) gives 205 blind, and pins them to exactly two structural causes: no operand
  identity, and no check of the operation's requirement against tracked state (gaps §14). The two
  numbers disagree for a reason worth a section: the fuzzer weights every ordering equally,
  including ones no sane kernel writes.
- Real bugs. Auditing shipped LLK code against F3 (`uninit` should undo `init`) found 6 confirmed
  gaps out of 44 init/uninit pairs, and a second pass on sticky config bits found 1 more, across
  three architectures (gaps §11, §12). "Not a true inverse" is common; "must be and is not" is rare,
  and we can say which is which.

**Skills this needs.** State machines, register field tables, reading LLK and Sanitizer source, and
a lot of careful counting. The contract is an FSM over a register map, which is native ground for a
digital designer; "typestate" is the name the software literature gives it, and is dressing on top.

**Where it sits in the literature.** Three lines meet here, and the paper should say so on page 2.

- *The rules are a typestate problem.* Strom and Yemini (IEEE TSE 1986) named it; DeLine and
  Fähndrich (ECOOP 2004) made it a per-object protocol machine; Aldrich et al. (TOPLAS 2014) gave
  the modern formal write-up. Our §6 table is a typestate automaton, one per execution unit. Say
  also why we did not use Design by Contract pre/post conditions (Meyer 1992; Hatcliff et al.,
  Computing Surveys 2012): pre/post pins down one call on its own, and our rules are about call
  *order*.
- *Vendors turning an informal manual into maths.* This is the closest precedent for "the spec
  existed only as prose, so we formalized it," and it is a strong frame because every large vendor
  has now done it. ARM: Reid (FMCAD 2016) and Reid, "Who Guards the Guards?" (OOPSLA 2017), which
  formalizes 59 prose properties and model-checks them against ARM's own machine-readable spec.
  Intel/AMD x86-64: Dasgupta, Park, Kasampalis, Adve and Roşu (PLDI 2019), a complete executable
  semantics for 3155 instruction variants that found bugs in the reference manual itself. NVIDIA:
  Lustig, Sahasrabuddhe and Giroux (ASPLOS 2019) on the PTX memory model. AMD and IBM: the ACL2 line
  (Russinoff et al. FMCAD 2000; Sawada et al. on Power4 divide/sqrt). Cross-vendor: Sail (Armstrong
  et al., POPL 2019). Our contribution is the same move one layer up: not the ISA, but the kernel
  library that sits on it, where the state is config registers rather than architectural registers.
- *Hardware/software contracts as a named object.* Guarnieri, Köpf, Reineke and Vila (IEEE S&P 2021)
  coined the term for secure speculation; Mosier, Lachnitt, Nemati and Trippel (ISCA 2022) gave the
  axiomatic form. Our `Tracked` and `Scratch` split is the same shape of statement: what software
  may assume, and what it must not.
- *Checkers of this kind.* AddressSanitizer (Serebryany et al., USENIX ATC 2012) is the model for
  "a dynamic checker that finds real bugs and ships." Runtime verification gives the vocabulary for
  monitors and monitorability (Bartocci, Falcone, Francalanza and Reger, LNCS 10457, 2018). For
  accelerators specifically, AccelSync (arXiv:2605.07881, 2026) checks synchronization coverage
  across DMA/vector/matrix units on shared buffers and finds hazards a runtime sanitizer misses.
  It is the nearest neighbour and the right one to compare against: it does cross-unit *data*
  ordering, we do per-unit *configuration* state. Different axis, same machine.

**What a reviewer will push on.** "Is this just a bug-finding tool paper?" The defence is the
blind-spot measurement. Nobody in the sanitizer literature reports what their tool structurally
cannot see, as a number, with two independent derivations. That metric, `certainty = 1 - blind
sites / state-changing calls`, is the part to sell.

**Venue shape.** A tool-and-measurement paper. ISPASS, CGO, or an OOPSLA/ASPLOS experience track.

**Status.** Mostly written between the two existing docs.

---

## Paper 2: HAL, fixing the register layer

**One sentence.** A typed C++ layer over the config registers turns a family of silent corruption
bugs into compile errors, and we measure the win the modern way: how much it costs an AI agent to
write a correct kernel with it and without it.

**The problem.** Today a kernel programs config registers by copying a raw address, a mask and a
shift out of a header. Three things go wrong.

- The constants are copied by hand, so they rot, and nobody can audit a kernel by reading it.
- Nothing is checked. A write to the wrong section or with an oversized value compiles.
- Config words alias. Two differently-named fields share one 32-bit word, so a read-modify-write of
  one silently clobbers the other, including one another *thread* just set. That last case is the
  bug class that costs weeks.

**What we do.** The Blackhole HAL (docs at `http://tensix-l-01:8785/docs/hal/index.html`). A field is
a generated descriptor (register file, address, section, shift, mask) and an access policy (MMIO,
config unit, scalar unit), both passed as template arguments. A kernel names a field and a value.
It never writes an address, a mask or a shift. `set()` groups assignments by physical word and emits
one update per word, with explicit ordering barriers. The layer is split along the same unit lines as
the contract: `cfg`, `address_counters`, `math_counters`, `gpr_ops`, `move`, `atomic`, `mop`,
`replay`, `sync`, `unpack`, `fpu`, `src`, `dst`, `pack`, `nop`. The flagship is `hal::cfg`.

**The claim we can be wrong about.** The config-word clobber class is **unrepresentable**, not
merely detectable. Grouping by physical word plus compile-time mask checking means "two fields, one
word, silent clobber" is a compile error. Tie it to paper 1: HAL is the layer that implements the
`g_φ` parameter encodings of formal §3 and §4, and it makes them typed.

**How we show it.** Two measurements, and the second is the one that will get the paper read.

1. *Hazards closed.* Take the hazard classes we catalogued. For each, say whether it is even
   *expressible* in raw style and in HAL style. This is a static argument, not a benchmark.
2. *Agent tokenomics.* Have an LLM agent write the same set of kernels twice, once against raw
   `ckernel`/cfg and once against HAL, and report cost per *correct* kernel: total tokens, number of
   compile-and-retry cycles, pass rate at k attempts, plus lines of kernel code and how many raw
   magic constants survive.

   The reason this is a real result and not a gimmick: an API's cost is now partly the price of
   getting a machine to use it correctly, and that price is measurable in a way human ergonomics
   never was. An API that needs a human to remember a mask is an API that burns tokens on retries.
   Report it as money and as pass@k, not as a vibe.

   Be honest about the threats to validity, because reviewers will go straight there: model version
   drift, contamination (HAL is new, raw ckernel is old and may be in training data, which biases
   *against* HAL and is worth saying out loud), prompt sensitivity, and the small size of the kernel
   set. Fix the model and the seed, publish the prompts and the harness, and report variance across
   repeats.

**Skills this needs.** Reading a register map and a C++ header, running an agent harness, and
designing a fair paired experiment. The HAL already exists, so this is a measurement paper about
something someone else built, not a build. The C++ template machinery has to be understood well
enough to explain, not written.

**Where it sits in the literature.**

- *Register descriptions as a standard.* IP-XACT (IEEE 1685-2022) and Accellera SystemRDL 2.0 give
  exactly this information (register, field, access policy) but from an *external* file that
  generates views. HAL's move is to bring it into the language, so the check happens in the type
  system of the code you are already writing.
- *Driver formalisms, the closest published match.* Devil (Mérillon et al., OSDI 2000) is an
  interface language for memory-mapped registers. Termite (Ryzhyk et al., SOSP 2009) goes further
  and synthesizes the driver from a register-plus-behaviour spec. Both tie an API call to register
  state, which is our exact subject.
- *Type-safe register access in practice.* The Tock register interface wraps each memory-mapped
  address in its own type exposing only the operations that address supports, and Rust embedded
  work generally encodes peripheral state in types with no runtime cost. This is the industrial
  precedent that HAL is not exotic. It also sets up paper 3, which does the same trick for
  *sequences* rather than single writes.
- *LLM code generation for accelerators, which is where the measurement comes from.* KernelBench
  (Ouyang, Guo, Arora, Zhang, Hu, Ré and Mirhoseini, arXiv:2502.10517, ICML 2025) is the reference
  for how to evaluate model-written GPU kernels, including the `fast_p` metric that scores
  correctness and speed together; its headline is that frontier models beat a PyTorch baseline in
  under 20% of cases. "Can Large Language Models Write Parallel Code?" (arXiv:2401.12554) gives 420
  tasks over seven execution models and the same conclusion for parallel code. AutoAPIEval
  (arXiv:2409.15228) is the framework for evaluating *API-oriented* generation specifically, which is
  what we are doing. And AccelSync reports a 19.2% defect rate on 120 LLM-generated accelerator
  kernels, which is the number that says this problem is worth an API change.
- *The older question underneath.* API usability research asked which interfaces humans get wrong.
  We are asking which interfaces *machines* get wrong, with a cost function attached.

**What a reviewer will push on.** "Your agent experiment measures the model, not the API." Answer
with the paired design (same model, same prompts, same kernels, only the API changes), variance
across repeats, and the contamination direction, which runs against us. Second push: "compile-time
checking of register writes is known." Answer: yes, from an external spec (IP-XACT, SystemRDL,
Devil); the contribution is in-language plus the word-grouping result, and the measurement is new
regardless.

**Venue shape.** A tools or experience paper. CGO, LCTES, or an AI-for-code venue if the agent
measurement leads.

---

## Paper 3: open

**What we actually want.** A reference manual for the part, in the sense of ST's RM0008 for the
STM32F10x family: the programmer's manual, block by block, every register and every field, with the
prose that says what they do and in what order to touch them. That is the right artifact, and it is
the thing this chip has never had.

**Why it is not paper 3.** It is too much work for one Tensix generation, let alone three. It cannot
be wrapped up in a couple of months, and it should not be written up as a proposal or a partial. It
gets published when it is finished.

So paper 3 is **open**. Papers 1 and 2 do not depend on it and should go ahead.

**Framings considered and set aside.** Recorded so they are not re-proposed.

1. *Static state tracking in the C++ type system* (the original plan). Set aside twice over: it is a
   compiler paper and the candidates are digital-design people, and its performance half was
   published at ASPLOS 2026 as the Configuration Wall paper. Kept below as paper 4 for whoever
   bridges into it later, because the correctness half is still unpublished and still good.
2. *Hardware characterization*: configuration cost, redundancy rate, and an ownership census over
   the config fields. Set aside.
3. *A framework or tool for generating reference manuals for custom ASICs.* Set aside, because what
   is wanted is the manual, not a framework for producing manuals. The gap is real and is named in
   the RDL literature (Black and Smith, DVCon: register interactions, access ordering, exclusivity
   and time ordering are requirements that IP-XACT, SystemRDL and UVM only partly meet, see
   related-work §4), and it stays on file as a thing to say *inside* the manual effort rather than
   instead of it.

**What stays true whatever paper 3 becomes.**

- The manual is the real goal, and a surprising amount of the material for it already exists: the
  state-audit map (5463 effect rows, 1479 config-register writes classed "retained until
  reconfigured"), the HAL generated field descriptors, paper 1's contract and FSM, and the
  measurements in the gaps doc. Whatever paper 3 turns out to be, it should fall out of building the
  manual rather than pull effort away from it.
- The constraint on paper 3 is fixed and worth restating: a first paper for a digital-design
  candidate, finishable in months, not a reinvention of something already published.
- The honest sizing question to answer first is what a full register-and-field pass over one Tensix
  generation actually costs in person-months. Until that number exists, every plan here is a guess.

**Open, to settle with the candidate:** what is the smallest piece of the manual effort that stands
on its own as a first paper.

---

## Paper 4, later: static state tracking, once the candidate has bridged

Not for a first paper, and not for someone who has not written a compiler pass. Parked here so the
work is not lost, and because papers 1 and 3 build its foundations: paper 3's layer 3 is the
ownership census this paper's central claim turns on, and its layer 4 is the per-operation read
dependency map a compile-time checker needs in order to exist at all. Whoever writes paper 3 will
have spent a year assembling exactly the facts this paper consumes, which makes them the right person
to write it next.

**Where this used to be.** An early plan had this as paper 3. The idea was "carry the contract state in C++ types, elide
redundant reconfigures, force a reconfigure when the state is unknown." Two problems with that as
written:

1. **It is published.** "The Configuration Wall" (ASPLOS 2026) is an MLIR abstraction whose passes
   are named State Tracing, Configuration Deduplication and Configuration Overlap, for a 2x geomean
   speed-up. Leading with compile-time elision of redundant configuration is now reinvention.
2. **Every ingredient is old.** Typestate in a type system runs from Strom and Yemini (1986) through
   Plaid and Rust to SquirrelFS (OSDI 2024), which uses Rust's typestate so that compiling *is* the
   crash-consistency proof. The known/unknown lattice with a widening merge is constant propagation
   (Wegman and Zadeck, TOPLAS 1991) over an abstract domain (Cousot and Cousot, POPL 1977). The loop
   fixed point in `control.h` is standard dataflow analysis.

**What survives, and it is a good paper for whoever writes it.** The claim that is not published and
not old: **configuration deduplication is unsound without an ownership contract.** A
reaching-definitions argument that deletes a write because the field already holds the value assumes
one writer. On a core where three threads configure shared state, that assumption can be false, and
the failure is silent wrong numbers. The ASPLOS work is single-control-thread and makes a performance
claim, not a correctness one. So the paper is: state the soundness condition, show a single-thread
analysis deleting a write it must keep, and show that the same lattice that does the deletion proves
the survivors sufficient, because "unknown" forces a reconfigure.

Its evidence comes from elsewhere in this programme: paper 3's ownership census says whether the
unsoundness is reachable at all, paper 1's 584 fuzzer scenarios and 7 confirmed bugs are the
correctness benchmark (which of them fail to compile?), and the prototype is tt-metal PR #49724
(`experiments/static-state-tracking/`: `inc/tracked.h`, `inc/state.h`, `inc/control.h`, Blackhole
only). The cross-thread part is not built yet and is the research to do.

Model to imitate: SquirrelFS. Position against: the ASPLOS paper, on three axes, namely shared state
across concurrent threads, a contract to be sound against, and no compiler changes.

---

## Notes on the set

- **Order.** Paper 1 first; everything else cites its contract, its bug catalogue and its fuzzer
  corpus. Paper 2 is independent and can run alongside it. Paper 3 is open, and paper 4 waits for
  someone who has bridged into compiler work.
- **The thing to settle before planning further.** What a full register-and-field pass over one
  Tensix generation costs in person-months. The reference manual is the real goal, and until that
  number exists, both the manual schedule and paper 3's scope are guesses.
- **Naming.** PR #49724 calls paper 4's subject "static state tracking" (SST). Earlier drafts called
  it AST/MST. Pick SST, and define it against the other names in the first paragraph.
- **What is grounded and what is inferred.** The contract, the bug counts, the blind-spot estimates
  and the fuzzer results are measured and live in this repo. Paper 2 is read from the HAL docs.
  Paper 4's mechanism is read from `tracked.h`, `state.h` and `control.h`, and its cross-thread part
  is *not built yet*. Every measurement plan above is a proposal, and paper 3's are the ones most
  worth sanity-checking against what the simulator and the RTL can actually run.
- **The set sells as one argument: one source, many views.** That is Sail's position and
  SystemRDL's, and it is ours. The contract is written once (paper 1), and then it is the checker
  that runs at run time (paper 1), the types a kernel programs against (paper 2), the manual a
  hardware engineer reads (the long job), and eventually the thing the compiler proves (paper 4).
  Every view is generated from or checked against the same source, which is why none of them can
  drift. The seven confirmed bugs are the argument for why drift is not a theoretical worry.
