# Related work: how others tie API/function to SW/HW state

Background for framing the LLK formal contract (see [`llk-formal-contract.md`](./llk-formal-contract.md))
and the three papers planned on it (see [`papers-plan.md`](./papers-plan.md)). There is no one
standard form.

**§1 to §4 frame the contract itself.** Four lines of work: the LLK contract is a mix of (1) and
(3), written in the style of (2), over the state model of (4). Anchors: Strom & Yemini (typestate),
DeLine & Fähndrich (object protocols), Guarnieri et al. (hardware-software contracts), Sail/ASL as
the executable-spec precedent, IP-XACT/SystemRDL for register-state description.

**For the reference manual, the long job, read §4 and §2.** §4 is the register description
standards, which cover the register tables well and, per Black & Smith, openly do not cover ordering,
ownership or exclusivity. §2 is the vendor precedent: everyone who converted a prose manual into a
machine-readable spec, and found bugs in the manual doing it. The asymmetry worth remembering is that
those teams all had a manual to convert, and we do not.

**§5 to §9 are the rest.** §6 is accelerator configuration cost and cross-unit checking: the ASPLOS
2026 configuration-wall paper and AccelSync, which between them say why configuration state is now
both the performance and the correctness bottleneck. §5 is the compiler machinery paper 4 must cite
so it is not accused of reinventing it. §7 is paper 1's company: dynamic checkers, and vendors
validating a prose spec against the machine. §8 and §9 are paper 2: typed register access in shipping
systems, and how to measure an API by what it costs a model to use it correctly.

---

## 1. Software side: which calls are legal depends on state (typestate)

The right name for the LLK §6 lifecycle FSM: the set of valid calls on an object is a function of
its current state, checked at compile time. The `INITIAL → CONFIGURED → INITIALIZED → EXECUTED →
UNINITIALIZED` machine *is* a typestate automaton.

- **R. E. Strom, S. Yemini, "Typestate: A Programming Language Concept for Enhancing Software
  Reliability," *IEEE TSE* 12(1), 1986.** Where the idea starts.
  <https://ieeexplore.ieee.org/document/6312929/> · <https://dblp.org/rec/journals/tse/StromY86.html>
- **R. DeLine, M. Fähndrich, "Typestates for Objects," *ECOOP* 2004.** Typestate as a per-object
  protocol FSM, the closest match to LLK's per-EXU FSM. See also "Enforcing High-Level Protocols in
  Low-Level Software," *PLDI* 2001.
  <https://link.springer.com/chapter/10.1007/978-3-642-22655-7_2> (empirical study citing this line)
- **J. Aldrich et al., "Foundations of Typestate-Oriented Programming," *ACM TOPLAS* 36(4), 2014.**
  The modern formal write-up; the Plaid language.
  <https://dl.acm.org/doi/10.1145/2629609> · overview: <https://en.wikipedia.org/wiki/Typestate_analysis>
- **K. Bierhoff, J. Aldrich, "Modular Typestate Checking of Aliased Objects," *OOPSLA* 2007.**
  Typestate under aliasing, useful if two kernels share operand/CB identity (cf. gaps §14).

### The path the LLK contract turned down: Design by Contract / pre-post

Worth citing to say why typestate fits and per-call pre/post does not: pre/post pins down one call
on its own, typestate pins down legal *sequences*. (The LLK doc bans `@pre`/`@post` for this reason;
the contract is a rule about call order, not a per-call Hoare triple.)

- **B. Meyer, "Applying 'Design by Contract'," *IEEE Computer* 25(10), 1992** (Eiffel).
- **J. Hatcliff, G. T. Leavens, K. R. M. Leino, P. Müller, M. Parkinson, "Behavioral Interface
  Specification Languages," *ACM Computing Surveys* 44(3), 2012.** Survey of JML, Larch, Spec#, and
  the Floyd–Hoare pre/post/frame line.
  <https://www.dcc.fc.up.pt/~nam/resources/VP2024/Hatcliff-et-al.---2012---Behavioral-interface-specification-languages.pdf>

---

## 2. Hardware side: entry points as executable state transformers (ISA as spec)

LLK §4 ("the five entry points as state transformers") is written in the spirit of vendor
executable ISA specs, where each op is a function over machine state.

- **A. Armstrong, T. Bauereiss, B. Campbell, A. Reid, K. E. Gray, R. M. Norton, P. Mundkur,
  M. Wassell, J. French, C. Pulte, S. Flur, I. Stark, N. Krishnaswami, P. Sewell, "ISA Semantics for
  ARMv8-A, RISC-V, and CHERI-MIPS," *POPL* 2019 (PACMPL 3).** The **Sail** ISA language; each
  instruction is a state transformer, with dependent bitvector types.
  <https://dl.acm.org/doi/abs/10.1145/3290384> · repo: <https://github.com/rems-project/sail> ·
  RISC-V model: <https://github.com/riscv/sail-riscv>
- **A. Reid, "Trustworthy Specifications of ARM v8-A and v8-M System Level Architecture," *FMCAD*
  2016**, and ARM's machine-readable **ASL**. The industry case for one authoritative executable
  spec built from vendor pseudocode.
  <https://alastairreid.github.io/ARM-v8a-xml-release/> · "State of Sail":
  <https://alastairreid.github.io/papers/SpISA_19/>

---

## 3. The HW/SW boundary contract: naming the split (Tracked vs Scratch)

LLK §4–§5 draw a line between state the caller may rely on and state each op sets fresh. That is a
hardware-software contract: what SW may assume about HW state, and what HW promises. LLK's
`Tracked ⊎ Scratch` split mirrors the observe/leak split below.

- **M. Guarnieri, B. Köpf, J. Reineke, P. Vila, "Hardware-Software Contracts for Secure
  Speculation," *IEEE S&P* 2021** (Best Paper). Coined the term; a contract fixes the SW-visible
  effects of HW execution.
  <https://www.microsoft.com/en-us/research/publication/hardware-software-contracts-for-secure-speculation/>
  · <https://arxiv.org/abs/2006.03841>
- **N. Mosier, H. Lachnitt, F. Nemati, C. Trippel, "Axiomatic Hardware-Software Contracts for
  Security," *ISCA* 2022.** The axiomatic form of the same idea.
  <https://dl.acm.org/doi/10.1145/3470496.3527412>

### Vendor precedent: turning the informal vendor doc into math (AMD / NVIDIA / IBM)

- **NVIDIA. D. Lustig, S. Sahasrabuddhe, O. Giroux, "A Formal Analysis of the NVIDIA PTX Memory
  Consistency Model," *ASPLOS* 2019.** Turns the English PTX spec into an axiomatic model
  (Alloy + Coq).
  <https://research.nvidia.com/index.php/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model>
  · <https://dl.acm.org/doi/10.1145/3297858.3304043>
- **AMD. D. M. Russinoff, M. Kaufmann, E. Smith, R. Sumners, "A Case Study in Formal Verification
  of Register-Transfer Logic with ACL2: The Floating-Point Adder of the AMD Athlon Processor,"
  *FMCAD* 2000**, plus Russinoff's ACL2 checks of AMD-K7/K5 FP RTL against IEEE 754.
  <https://link.springer.com/chapter/10.1007/3-540-40922-X_3>
- **IBM. ACL2 check of the Power4 floating-point divide/square-root** (J. Sawada et al.). Same ACL2
  line used across AMD, Centaur, IBM, Rockwell Collins.
  <https://www.cs.utexas.edu/users/moore/acl2/manuals/current/manual/index-seo.php/ACL2____INTERESTING-APPLICATIONS>

---

## 4. Register / machine-state description standards (LLK §2 field tables)

For a standards citation for "here is the machine state, field by field, with an access policy."
LLK's `y_U`/`y_P` operand-field sets and `x_j` own-field sets are a hand-rolled version of these.

- **IEEE 1685 (IP-XACT), "Standard Structure for Packaging, Integrating, and Reusing IP within Tool
  Flows," IEEE Std 1685-2022.** Machine-readable register/field/memory-map plus access policy.
  <https://ieeexplore.ieee.org/document/10054520/> · guide: <https://www.arteris.com/learn/ip-xact/>
- **Accellera SystemRDL 2.0.** One source for a register description; every view (RTL, docs, model)
  is generated from it.
  <https://www.accellera.org/activities/working-groups/systemrdl>
- **D. C. Black, D. Smith (Doulos), "Comprehensive Register Description Languages: The Case for
  Standardization of RDLs Across Design Domains," *DVCon*.** **The citation paper 3 opens with.**
  Surveys the RDL landscape and lists the requirements no standard meets. Among them: item 8, "allow
  description of interactions between registers (e.g. access ordering requirements and exclusivity)",
  and item 10, "specify time ordering aspects (one register must be read before the second)". In
  their words: "There are also problems with relationships among registers. Often there are
  dependencies on the order of access or interactions between configurations/status of different
  registers. These need to be expressed and understood by all the design domains." They conclude that
  IP-XACT, SystemRDL and UVM give "partial solution to this problem area." Paper 3's layers 2 to 5
  are an attempt at items 8 and 10 for a real part.
  <https://dvcon-proceedings.org/wp-content/uploads/comprehensive-register-description-languages.pdf>

### Device-driver formalisms: register state paired with a call protocol

The tightest published match to "an API function acting on hardware register state," including
generating the driver from the spec.

- **F. Mérillon, L. Réveillère, C. Consel, R. Marlet, G. Muller, "Devil: An IDL for Hardware
  Programming," *OSDI* 2000.** An interface language for memory-mapped registers and ports.
  <https://www.semanticscholar.org/paper/Devil:-an-IDL-for-hardware-programming-M%C3%A9rillon-R%C3%A9veill%C3%A8re/35de016d68726affed62edcf35eefcfc3ee30b27>
- **L. Ryzhyk, P. Chubb, I. Kuz, E. Le Sueur, G. Heiser, "Automatic Device Driver Synthesis with
  Termite," *SOSP* 2009.** Feeds a register-plus-behaviour spec and an OS-interface spec into driver
  synthesis: a formal spec of exactly the API-to-register tie.
  <https://www.sigops.org/s/conferences/sosp/2009/papers/ryzhyk-sosp09.pdf>

---

## 5. Compile-time state analysis: the classical machinery paper 4 stands on

For paper 4, not paper 3. The known/unknown lattice, the widening `merge`, and the loop fixed point
in `tracked.h`/`control.h` are textbook dataflow analysis moved into the type system. Cite these up
front so the paper is not accused of reinventing them.

- **P. Cousot, R. Cousot, "Abstract Interpretation: A Unified Lattice Model for Static Analysis of
  Programs by Construction or Approximation of Fixpoints," *POPL* 1977.** The lattice, the abstract
  domain, and widening.
  <https://dl.acm.org/doi/10.1145/512950.512973> · <https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml>
- **M. N. Wegman, F. K. Zadeck, "Constant Propagation with Conditional Branches," *ACM TOPLAS*
  13(2), 1991.** The known/unknown/⊥ domain and the branch merge, which is exactly `Tracked<T>` and
  `merge`.
  <https://dl.acm.org/doi/abs/10.1145/103135.103136> ·
  <https://www.cs.utexas.edu/~pingali/CS380C/2010/papers/p291-wegman.pdf>
- **J. Knoop, O. Rüthing, B. Steffen, "Lazy Code Motion," *PLDI* 1992.** Optimal partial redundancy
  elimination; the reference point for "we deleted the redundant work."
  <https://dl.acm.org/doi/10.1145/152819.152823> (variation) ·
  <https://homepages.dcc.ufmg.br/~fernando/classes/dcc888/ementa/slides/LazyCodeMotion.pdf>
- **W. Taha, T. Sheard, "MetaML and Multi-Stage Programming with Explicit Annotations,"
  *Theoretical Computer Science* 248(1-2), 2000.** The principled account of computing at compile
  time with type safety across stages. C++ templates are the informal version.
  <https://www.sciencedirect.com/science/article/pii/S0304397500000530>
- **H. LeBlanc, N. Taylor, J. Bornholt, V. Chidambaram, "SquirrelFS: Using the Rust Compiler to
  Check File-System Crash Consistency," *OSDI* 2024.** The model to imitate: typestate in a real
  type system replaces a separate proof, so successful compilation *is* the guarantee.
  <https://www.usenix.org/conference/osdi24/presentation/leblanc> ·
  <https://www.usenix.org/system/files/osdi24-leblanc.pdf>

---

## 6. Accelerator configuration overhead, and cross-unit checking

Paper 4's home section, and the evidence paper 3 uses for "why this matters now.".

- **J. Van Delm, A. Lydike, J. Dumoulin, J. Crols, X. Yi, R. Antonio, J. Woodruff, T. Grosser,
  M. Verhelst, "The Configuration Wall: Characterization and Elimination of Accelerator
  Configuration Overhead," *ASPLOS* 2026.** A configuration roofline model plus an MLIR abstraction
  (`accfg`) whose passes are State Tracing, Configuration Deduplication and Configuration Overlap.
  2x geomean on OpenGeMM, plus Gemmini results. Two uses. **For paper 3** it is one half of the
  "why now": configuration overhead is a first-order performance limit, so the state nobody documents
  is also the state that costs the most. **For paper 4** it is the competition, because it is the
  published version of paper 4's perf claim. Note what it is not: a compiler IR pass on
  single-control-thread accelerators, claiming performance, not correctness.
  <https://doi.org/10.1145/3760250.3762225> · PDF: <https://antonlydike.de/publications/asplos26-accfg.pdf>
  · <https://www.research.ed.ac.uk/en/publications/the-configuration-wall-characterization-and-elimination-of-accele/>
- **H. An, R. Wang, D. Qian, "AccelSync: Verifying Synchronization Coverage in Accelerator Pipeline
  Programs," arXiv:2605.07881, 2026.** Formalizes accelerator pipeline programs with program,
  synchronization and barrier order, and reduces correctness to barrier sufficiency. 6292 production
  CANN kernels (3 unknown hazards), 120 LLM-generated kernels (19.2% defect rate), beats Huawei's
  msSanitizer at 400x lower per-kernel cost. Cross-unit ordering on **data buffers**; we do
  **configuration state**. Their line that cross-unit hazards "escape both simulation and golden
  testing" is the other half of paper 3's "why now", and the 19.2% figure is paper 2's.
  <https://arxiv.org/abs/2605.07881>
- **A. Betts, N. Chong, A. F. Donaldson, S. Qadeer, P. Thomson, "GPUVerify: A Verifier for GPU
  Kernels," *OOPSLA* 2012.** Race and divergence freedom by reduction to a sequential program.
  The precedent for verifying many-threaded kernels without reasoning about interleavings.
  <https://dl.acm.org/doi/10.1145/2384616.2384625> · journal version, *TOPLAS* 2015:
  <https://dl.acm.org/doi/10.1145/2743017>
- **K. Honda, N. Yoshida, M. Carbone, "Multiparty Asynchronous Session Types," *POPL* 2008.** Typing
  a protocol among several parties. The tool of choice if the cross-EXU gap (Q2, G4b) is ever closed.
  <https://dl.acm.org/doi/10.1145/1328438.1328472>

---

## 7. Dynamic checkers: the company paper 1's Sanitizer keeps

- **K. Serebryany, D. Bruening, A. Potapenko, D. Vyukov, "AddressSanitizer: A Fast Address Sanity
  Checker," *USENIX ATC* 2012.** The model for a dynamic checker that finds real bugs and ships.
  <https://www.usenix.org/conference/atc12/technical-sessions/presentation/serebryany> ·
  <https://www.usenix.org/system/files/conference/atc12/atc12-final39.pdf>
- **E. Bartocci, Y. Falcone, A. Francalanza, G. Reger, "Introduction to Runtime Verification,"
  in *Lectures on Runtime Verification*, LNCS 10457, Springer 2018.** Vocabulary for monitors,
  instrumentation, and monitorability, which is the formal name for paper 1's blind-spot question.
  <https://link.springer.com/chapter/10.1007/978-3-319-75632-5_1>

### Vendor specs, expanded (companion to §3)

- **A. Reid, "Who Guards the Guards? Formal Validation of the Arm v8-M Architecture Specification,"
  *OOPSLA* 2017.** 59 prose properties from the architecture manual, model-checked against ARM's own
  machine-readable spec. The closest precedent for "the rules existed only as prose, so we wrote
  them down and checked them."
  <https://dl.acm.org/doi/10.1145/3133912> ·
  <https://alastairreid.github.io/papers/oopsla2017-whoguardstheguards.pdf>
- **S. Dasgupta, D. Park, T. Kasampalis, V. S. Adve, G. Roşu, "A Complete Formal Semantics of x86-64
  User-Level Instruction Set Architecture," *PLDI* 2019.** 3155 instruction variants, executable,
  and it found bugs in the reference manual.
  <https://dl.acm.org/doi/10.1145/3314221.3314601> ·
  <https://fsl.cs.illinois.edu/publications/dasgupta-park-kasampalis-adve-rosu-2019-pldi.pdf>

---

## 8. Typed register access in shipping systems (paper 2's industrial precedent)

Companion to §4, which covers the external-spec standards. These bring the check into the language,
which is HAL's move.

- **Tock register interface.** Each memory-mapped address is wrapped in its own type that exposes
  only the operations that address supports (`ReadWrite`, `ReadOnly`, `WriteOnly`), with a macro DSL
  that reads like a datasheet. Zero runtime cost.
  <https://www.tockos.org/blog/2018/mmio-registers/> · <https://docs.rs/tock-registers/latest/tock_registers/>
- **The Embedded Rust Book, "Typestate Programming."** Peripheral state encoded in zero-sized types,
  so an invalid sequence fails to compile. The practitioner's version of §1.
  <https://docs.rust-embedded.org/book/static-guarantees/typestate-programming.html>

---

## 9. Measuring an API by what it costs a model to use (paper 2's evaluation)

- **A. Ouyang, S. Guo, S. Arora, A. L. Zhang, W. Hu, C. Ré, A. Mirhoseini, "KernelBench: Can LLMs
  Write Efficient GPU Kernels?" arXiv:2502.10517, *ICML* 2025.** 250 PyTorch workloads, and the
  `fast_p` metric that scores correctness and speed-up together. Frontier reasoning models match the
  PyTorch baseline in under 20% of cases. The methodological reference for paper 2's experiment.
  <https://arxiv.org/abs/2502.10517> · <https://github.com/ScalingIntelligence/KernelBench>
- **"Can Large Language Models Write Parallel Code?" arXiv:2401.12554.** 420 tasks over serial,
  OpenMP, Kokkos, MPI, MPI+OpenMP, CUDA and HIP (the ParEval benchmark), with metrics for
  performance and parallel scaling.
  <https://arxiv.org/html/2401.12554v3.pdf> · repository-level follow-on, *ICPP* 2025:
  <https://dl.acm.org/doi/10.1145/3754598.3754669>
- **"A Comprehensive Framework for Evaluating API-oriented Code Generation in Large Language
  Models" (AutoAPIEval), arXiv:2409.15228.** Evaluating generation *against a specific library API*,
  which is exactly the raw-cfg versus HAL comparison.
  <https://arxiv.org/abs/2409.15228>
- **AccelSync (§6) reports a 19.2% defect rate on 120 LLM-generated accelerator kernels.** The single
  number that says an accelerator API's machine-usability is a correctness problem, not a comfort one.

---

*Notes:* citation details (authors, venue, year) are copied from the sources linked above; check
page and DOI against the venue before you publish. Every claim here traces to a linked source, not
to memory.
