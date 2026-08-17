# Related-Work Sweep (Web): Novelty Assessment for "Exact Top-K Selection on a Dataflow Many-Core"

Date: 2026-08-16. Scope: web search (arXiv, ACM/IEEE, vendor docs) for prior art against contributions C1–C4 plus the agentic-methodology section. No repo edits, no device runs.

## Verdict summary

| Contribution | Verdict | Deciding prior art |
|---|---|---|
| C1 — GPU radix-select economics don't transfer (measured negative result) | **SAFE** (with mandatory citations) | No prior work analyzes counting-pass/materialization economics of exact selection on a NoC-mesh dataflow many-core. Closest: TPU-KNN (NeurIPS'22) + "Faster Generalized Two-Stage Approximate Top-K" (arXiv:2506.04165) — they *dodge* exact selection via approximation, which supports C1's thesis. |
| C2 — log-tree column-parallel top-k on a mesh, no atomics/global sync, cost model, rectangle embedding | **NEEDS-REPOSITIONING** | 1990s theory: Krizanc & Narayanan, "Optimal algorithms for selection on a mesh-connected processor array" (SPDP'92) and "Fast deterministic selection on mesh-connected processor arrays" (Algorithmica '93, 1.45n steps). Reposition from "first tree top-k on a mesh" to "first *measured, cost-modeled* top-k system on commercial NoC-mesh silicon; theory assumed rank-selection on abstract 1-elt/PE meshes." |
| C3 — chunk-skip with running user-k threshold, soundness proof, rank-statistic skip law, calibrated forecast | **SAFE** (with mandatory citations) | Closest: WarpSelect (Johnson et al., running per-lane thresholds in bitonic queues) and Guess-Verify-Refine (arXiv:2604.22312, *predicted* threshold from temporal correlation, NVIDIA Blackwell). Neither has the rank-statistic skip law P≈e^(−K/(c+1)) → compile-time gate, soundness proof in a streaming bitonic cascade, or a pre-build no-go forecast. |
| C4 — first-public silicon characterization (packer exponent histogram, sign-magnitude total order, bf16 canonicalization, sync floors) | **NEEDS-REPOSITIONING** | "Dissecting the Tenstorrent Blackhole architecture via microbenchmarking" (2025, tech report hosted at asplos.dev; link now 404) + six Wormhole/Grayskull papers already characterize NoC/compute/memory performance. **None** covers packer numerics, NaN/subnormal canonicalization, sign-magnitude ordering, or top-k/sorting (verified against the citation graph of arXiv:2605.07599). Narrow the claim to "first public characterization of the Tensix *datapath numerics and sorting-relevant primitives*." |
| Methodology — agentic campaign | SAFE as a *section* (not a headline claim) | AccelOpt (arXiv:2511.15915), KForge, PEAK, AI CUDA Engineer, KernelBench exist for LLM-driven kernel work; an agentic *design-space-study campaign* narrative is still uncommon, but don't claim first-ness. |

---

## 1. Top-k / sorting on non-GPU AI accelerators (bears on C1, C2)

### Tenstorrent (most important — direct platform overlap)
- **Brown & Barton, "Accelerating stencils on the Tenstorrent Grayskull RISC-V accelerator"** (SC24 Workshops; arXiv:2409.18835). Stencil perf characterization on Grayskull Tensix. No selection/sorting, no numerics. Cite in C4 lineage.
- **"Assessing Tenstorrent's RISC-V MatMul Acceleration Capabilities"** (arXiv:2505.06085). Grayskull e75 matmul study, BF16 throughput/energy vs Xeon. No selection. C4 lineage.
- **Brown, Davies, Le Clair, "Exploring Fast Fourier Transforms on the Tenstorrent Wormhole"** (ISC 2025 Workshops; arXiv:2506.15437). FFT on Wormhole. C4 lineage.
- **"Stencil Computations on Tenstorrent Wormhole"** (arXiv:2605.07599). Stencils on Wormhole n300; measures kernel perf, energy, phase breakdowns. Its related-work section is the authoritative census of Tensix papers — **confirms none of the cited works characterize packer behavior, bf16 numerics, NaN handling, top-k, or sorting.**
- **"Numerical Kernels on a Spatial Accelerator: A Study of Tenstorrent Wormhole"** (arXiv:2603.23343). Three numerical kernels + CG solver vs NVIDIA GPUs; sparse-algorithm optimizations. No selection, no datapath numerics.
- **"Operator Fusion for LLM Inference on the Tensix Architecture"** (arXiv:2606.09879). Fusion strategy for data locality on Tensix. No selection.
- **Amati et al., "Accelerating Gravitational N-Body Simulations Using the RISC-V-Based Tenstorrent Wormhole"** (SC'25 Workshops; arXiv:2605.02744 / 2509.19294). Porting-strategy + perf study. No selection.
- **Vasiljevic & Capalija, "Blackhole & TT-Metalium: the standalone AI computer and its programming model"** (Hot Chips 36, 2024). Vendor architecture talk — the canonical Blackhole architecture citation.
- **"Dissecting the Tenstorrent Blackhole Architecture via Microbenchmarking"** (2025). Cited as ref [8] of arXiv:2605.07599 with URL `https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf` — **now 404**; asplos.dev is a personal site (Yiwei Yang, UCSC systems researcher), so this is an informal tech report, not a peer-reviewed ASPLOS paper. Reported findings (via secondary citations): SFPU sustains ~32 elem/cycle FP32 add; cache-free design doesn't hurt regular-access single-core perf but shifts burden to software; Mandelbrot 22.4x vs 1 CPU core. **This is the closest C4 threat**: it is a *general* Blackhole microbenchmark. It does not touch packer exponent histograms, canonicalization, sign-magnitude order, or count/rendezvous floors. Cite it, and narrow C4's first-ness to the numerics/sorting-primitive axis. Its instability (dead link, non-archival) is itself an argument for an archival characterization.
- Engineering artifacts, not publications: `ttnn.topk` docs, `ckernel::topk_local_sort` GitHub issues (#33492), sampling-op issue #16854. Cite as evidence the vendor stack routes top-k through a bitonic local sort, and that no perf study exists.

### Graphcore IPU
- **Jia, Tillman, Maggioni, Scarpazza, "Dissecting the Graphcore IPU Architecture via Microbenchmarking"** (Citadel tech report, arXiv:1912.03413, 2019). The genre template for C4-style work on a non-GPU AI chip; does not cover selection or numerics quirks.
- **Poplar/PopLibs `popops::TopK`** (API docs, v3.1.0). A production top-k library on a 1472-tile BSP many-core exists — but no published algorithm/perf paper found. Cite the docs; note absence of an economics study. IPU MD-simulation blog mentions QuickSort+InsertSort top-k as an application detail only.

### Cerebras WSE / Groq / SambaNova
- No sorting/selection/top-k publication found for any of them (searched: WSE sort/selection/top-k; Groq LPU top-k; SambaNova RDU sort). WSE papers are stencil/SpMM/LLM-benchmark (e.g., arXiv:2010.03660, 2210.04795, 2409.00287, 2605.07954, 2604.27985). **Gap confirms C1/C2 novelty space**: no exact-selection study exists on any 2D-mesh dataflow machine.

### TPU
- **Chern et al., "TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s"** (NeurIPS 2022). Approximate top-k via partial reduction shaped for the MXU; explicit accelerator-roofline argument that exact top-k is hostile to matmul-centric accelerators. **Mandatory C1 citation** — frames approximation as the industry escape hatch; C1 quantifies *why* the exact path fails on a NoC mesh.
- **"A Faster Generalized Two-Stage Approximate Top-K"** (arXiv:2506.04165, 2025; OpenReview izqZ1Crpjz). Generalizes two-stage approx top-k (top-1-per-bucket → sort); TPUv5e implementation. Same role as above.

### Academic many-cores (Manticore/Occamy/Snitch, Esperanto)
- **Zaruba, Schuiki, Benini, "Manticore: A 4096-Core RISC-V Chiplet Architecture"** (IEEE Micro 2021); **"Occamy: A 432-Core ... Dual-HBM2E RISC-V-Based Accelerator"** (ISSCC/JSSC line; arXiv:2501.07330). Dense/sparse FP workloads; **no sorting/selection benchmark found** on either. Esperanto ET-SoC-1: nothing on selection. Cite as "many-core selection remains unstudied" support.

**C1 verdict: SAFE.** The GPU radix-select lineage (below) has never been re-costed on a dataflow mesh; accelerator papers either avoid exact selection (TPU line) or don't treat it at all (WSE, IPU, Tenstorrent papers). C1's negative-result framing (narrow counting passes, 81-cycle decisions, materialization gap, degeneration to threshold bisection) has no published counterpart. Must-cite set: Alabi'12, Johnson'19 (WarpSelect), Gaihre'21 (Dr. Top-k), SC'23 comprehensive study, RadiK ICS'24, TPU-KNN, 2506.04165.

## 2. Merge-tree / tournament top-k on meshes; FPGA selection networks (bears on C2)

### Parallel-theory prior art (the decisive C2 citations)
- **Krizanc & Narayanan, "Optimal algorithms for selection on a mesh-connected processor array"** (IEEE SPDP 1992, DOI 10.1109/SPDP.1992.242761).
- **"Fast deterministic selection on mesh-connected processor arrays"** (Algorithmica, 1.45n steps on n×n mesh vs 2n+o(n) sort-based; Springer DOI 10.1007/BF01961542; earlier version LNCS, 10.1007/3-540-54967-6_79).
- **"Multi-packet selection on mesh-connected processor arrays"** (IPPS 1992): N≥p elements, N/p per PE — O(min(p·log(N/p), max(N/p^(2/3), √p))) communication steps. The multi-packet regime is exactly C2's ceil(C/P) regime.

These solve *rank selection* (k-th element) on abstract synchronous meshes with unit-cost neighbor communication. They do not: return the full top-k set with values+indices, model a real NoC (multicast, per-hop cost, DRAM ingress), handle atomics-free rendezvous on real silicon, or measure anything. **C2 must cite these and claim: first measured top-k (full set, values+indices) on commercial NoC-mesh silicon, with a validated cost model 2·ceil(C/P)+ceil(log2 P) and cost-optimal rectangle embedding.** The "without atomics/global sync" phrasing should be positioned against GPU practice (atomic counters in radix select), not against mesh theory (which never had atomics).

### Distributed DB top-k (conceptual tournament ancestors)
- **Fagin, Lotem, Naor, "Optimal aggregation algorithms for middleware"** (PODS'01/JCSS'03) — threshold algorithm (TA).
- **Cao & Wang, "Efficient top-K query calculation in distributed networks"** (PODC 2004) — TPUT, three-phase uniform threshold.
- **Michel, Triantafillou, Weikum, "KLEE: A Framework for Distributed Top-k Query Algorithms"** (VLDB 2005).
These merge per-node top-k lists under thresholds across a network — the tournament/merge-tree idea at datacenter granularity. Cite in C2 related work as a different cost regime (round-trips vs NoC hops).

### FPGA / hardware sorting-and-selection
- **Samardzic, Qiao, Aggarwal, Chang, Cong, "Bonsai: High-Performance Adaptive Merge Tree Sorting"** (ISCA 2020, DOI 10.1109/ISCA45697.2020.00033). Merge-tree sorting tuned to bandwidth/resource model — the closest *hardware merge-tree with an analytic cost model*; it sorts fully rather than selecting, on FPGA+DRAM rather than a mesh of cores. Top-2/3 closest for C2.
- **Papaphilippou et al., "FLiMS: a Fast Lightweight 2-way Merger for Sorting"** (arXiv:2112.05607). Building-block merger.
- **Qiao et al., "TopSort: A High-Performance Two-Phase Sorting Accelerator ... HBM"** (FCCM 2022; arXiv:2205.07991).
- **Jalilvand et al., "Sorting it out in Hardware: A State-of-the-Art Survey"** (arXiv:2310.07903). Survey incl. partial-sorting/top-k networks (even-odd swap + bitonic + parallel swap structures) — use as the umbrella citation for FPGA selection networks.
- **"Finding the Top-K Heavy Hitters in Data Streams: A Reconfigurable Accelerator Based on an FPGA-Optimized Algorithm"** (Electronics 12(11):2376, MDPI 2023). HLS streaming top-k heavy hitters.
- **Parravicini et al., "Scaling up HBM Efficiency of Top-K SpMV for Approximate Embedding Similarity on FPGAs"** (DAC 2021; arXiv:2103.04808). Top-k fused into SpMV for recommender similarity.
- **Sukhwani et al. / Casper & Olukotun** database-hardware line (top-k/sort for DB on FPGA) — optional depth.
- No hit for a *systolic top-k on a 2D mesh NoC of programmable cores*; mesh-of-trees NoC papers (Balkan et al., TVLSI 2008) are topology work, not selection algorithms.

**C2 verdict: NEEDS-REPOSITIONING** (survivable): the *idea* of tree-merged selection on a mesh exists in 90s theory and in FPGA merge trees; the *measured system contribution* (cost model validated to 104 cores, rectangle embedding, atomics-free rendezvous on shipping silicon) is unclaimed territory. The decisive citation to get ahead of a reviewer: Krizanc & Narayanan SPDP'92/Algorithmica'93.

## 3. Running-threshold pruning in streaming top-k (bears on C3)

- **Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs"** (IEEE Trans. Big Data 2019; arXiv:1702.08734). **WarpSelect**: per-lane thread queues with a running k-th-value threshold; elements failing the threshold are discarded without touching the bitonic merge. The single closest GPU mechanism to C3's chunk-skip. C3 differentiators: chunk-granular skip (not element-granular), soundness proof for skipping whole chunks in a *cascade* (partial state), closed-form skip law, compile-time gate.
- **"Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation"** (arXiv:2604.22312, 2026). Uses *previous decode step's* top-k as predicted threshold, secant-style counting refinement, bit-exact; 1.88x over production radix-select in TensorRT-LLM. Different signal (temporal vs within-stream rank statistics) and platform (GPU), but a reviewer will surface it — cite and contrast: GVR's threshold is a heuristic guess requiring verify/refine passes; C3's running threshold is *sound by construction* (no verification pass), and the skip law is distribution-free over random arrival orders.
- **Wang, Zhang, Han, "SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning"** (HPCA 2021; arXiv:2012.09852). Dedicated hardware top-k engine (quick-select threshold filter). Hardware-top-k-engine prior art; not streaming-skip.
- **"Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models"** (arXiv:2512.14661). Streaming accelerator with threshold-filtered top-k engine (chunked design with running top-m buffer + in-register bitonic merge per search snippets). Check full text before submission — closest *architecture-paper* analog of a chunked bitonic running-threshold engine.
- **AMD ROCm blog, "Adaptive Top-K Selection: Eliminating Performance Cliffs Across All K Values on AMD GPUs"** (2025/26). Ballot-based threshold filtering before sort. Grey literature; cite for completeness.
- Classic streaming: **Charikar, Chen, Farach-Colton** (Count-Sketch, ICALP 2002); **Metwally, Agrawal, El Abbadi** (SpaceSaving, ICDT 2005) — frequency-based top-k, different problem (heavy hitters ≠ order statistics); one-line related-work mention.
- Rank-statistic foundation: expected updates of a running top-k over a random-order stream is k·ln(n/k)+O(k) — records/left-to-right-maxima folklore (cite a textbook, e.g., Knuth TAOCP vol.3 or a records-theory survey). C3's P≈e^(−K/(c+1)) chunk-skip law is a chunked corollary; no publication found stating it or using it to derive a compile-time K/4 gate.
- Calibrated host-simulation forecast that no-go'd a variant pre-build: no analog found in the top-k literature; analogs live in the LLM-agent kernel papers (below) as "performance prediction," none calibrated against silicon for selection kernels.

**C3 verdict: SAFE**, contingent on explicitly contrasting WarpSelect and GVR in the paper. The soundness proof + distribution-free skip law + compile-time gate + forecast-driven variant elimination combination is unclaimed.

## 4. Tensix microarchitecture characterization (bears on C4)

Census (from section 1): 8 published items characterize Tensix *performance* (Grayskull stencils, matmul; Wormhole FFT, stencils, CG/numerical kernels, N-body, operator fusion; Blackhole informal microbenchmark). Verified via arXiv:2605.07599's bibliography: **none covers** packer behavior, exponent histogram sampling/aliasing, bf16 datapath canonicalization (NaN→Inf, −0→+0, subnormal→+0), sign-magnitude total order, or count/rendezvous synchronization floors.

Software prior art to acknowledge for the sign-magnitude point: the float→orderable-uint bit-flip trick (Herf, "Radix Tricks," 2001; Merrill & Grimshaw, "High Performance and Scalable Radix Sorting," Parallel Processing Letters 2011; used in CUB/Thrust) and IEEE 754-2019's totalOrder predicate. C4's novelty is that the Tensix *hardware datapath itself* imposes a sign-magnitude total order including NaN and canonicalizes bf16 specials — a silicon-behavior discovery, not a software encoding. State it that way.

**C4 verdict: NEEDS-REPOSITIONING** — from "first-public silicon characterization [of Blackhole]" to "first-public characterization of the Tensix compute datapath's numeric semantics and sorting-relevant primitive costs." The broad claim dies against the asplos.dev tech report and the Wormhole paper ecosystem; the narrowed claim is uncontested and stronger (it's also archival where the closest competitor is a dead link).

## 5. Agentic-campaign methodology (section positioning)

- **Ouyang et al., "KernelBench: Can LLMs Write Efficient GPU Kernels?"** (2025; arXiv:2502.10517) — benchmark.
- **Sakana AI, "The AI CUDA Engineer"** (2025 tech report) — agentic convert/translate/optimize/compose pipeline (and a famous reward-hacking cautionary tale — useful citation for why the paper's verification discipline matters).
- **"AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization"** (arXiv:2511.15915) — LLM agents on a *non-GPU accelerator* (Trainium-class); closest methodology cousin.
- **"KForge: Program Synthesis for Diverse AI Hardware Accelerators"** (arXiv:2511.13274; also arXiv:2606.02963 "KForge: LLM-Driven Cross-Platform Kernel Generation for AI Accelerators") — cross-platform, may include Tenstorrent as a target: **check its target list before submission.**
- **"PEAK: A Performance Engineering AI-Assistant for GPU Kernels"** (arXiv:2512.19018); **"EvoEngineer"** (arXiv:2510.03760); **Astra** (Stanford multi-agent GPU kernel optimization).
Position the methodology section as: prior agentic work optimizes *given* kernels against a benchmark; this campaign ran a *design-space study with measured no-gos* (forecast-driven variant elimination) on novel silicon. Do not claim "first agentic kernel work."

---

## Bibtex-ready citation list

```bibtex
% ---- GPU top-k lineage (C1) ----
@article{alabi2012kselection, author={Alabi, Tolu and Blanchard, Jeffrey D. and Gordon, Bradley and Steinbach, Russel}, title={Fast K-Selection Algorithms for Graphics Processing Units}, journal={ACM Journal of Experimental Algorithmics}, volume={17}, year={2012}, doi={10.1145/2133803.2345676}}
@article{johnson2019billion, author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}}, title={Billion-Scale Similarity Search with GPUs}, journal={IEEE Transactions on Big Data}, volume={7}, number={3}, year={2019}, note={WarpSelect. arXiv:1702.08734}}
@inproceedings{shanbhag2018topk, author={Shanbhag, Anil and Pirk, Holger and Madden, Samuel}, title={Efficient Top-K Query Processing on Massively Parallel Hardware}, booktitle={SIGMOD}, year={2018}, doi={10.1145/3183713.3183735}}
@inproceedings{gaihre2021drtopk, author={Gaihre, Anil and others}, title={Dr. Top-k: Delegate-Centric Top-k on GPUs}, booktitle={SC}, year={2021}, note={arXiv:2109.08219}}
@inproceedings{zhang2023paralleltopk, title={Parallel Top-K Algorithms on GPU: A Comprehensive Study and New Methods}, booktitle={SC}, year={2023}, doi={10.1145/3581784.3607062}, note={[verify authors]}}
@inproceedings{li2024radik, title={RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection}, booktitle={ICS}, year={2024}, doi={10.1145/3650200.3656596}, note={arXiv:2501.14336; [verify authors]}}
@article{xie2024rtopk, title={RTop-K: Ultra-Fast Row-Wise Top-K Selection for Neural Network Acceleration on GPUs}, year={2024}, note={arXiv:2409.00822; [verify authors/venue]}}
@article{gvr2026, title={Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation}, year={2026}, note={arXiv:2604.22312; [verify authors]}}
@misc{amd2025adaptivetopk, title={Adaptive Top-K Selection: Eliminating Performance Cliffs Across All K Values on AMD GPUs}, howpublished={ROCm Blogs}, url={https://rocm.blogs.amd.com/software-tools-optimization/adaptive-topk/README.html}}

% ---- Accelerator top-k / approximation (C1) ----
@inproceedings{chern2022tpuknn, author={Chern, Felix and Hechtman, Blake and Davis, Andy and Guo, Ruiqi and Majnemer, David and Kumar, Sanjiv}, title={TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s}, booktitle={NeurIPS}, year={2022}, note={[verify author list]}}
@article{twostage2025, title={A Faster Generalized Two-Stage Approximate Top-K}, year={2025}, note={arXiv:2506.04165; OpenReview izqZ1Crpjz; [verify authors]}}

% ---- Mesh selection theory (C2, decisive) ----
@inproceedings{krizanc1992optimal, author={Krizanc, Danny and Narayanan, Lata}, title={Optimal Algorithms for Selection on a Mesh-Connected Processor Array}, booktitle={IEEE SPDP}, year={1992}, doi={10.1109/SPDP.1992.242761}, note={[verify author list]}}
@article{meshselection_algorithmica, title={Fast Deterministic Selection on Mesh-Connected Processor Arrays}, journal={Algorithmica}, doi={10.1007/BF01961542}, note={1.45n steps on n x n mesh; [verify authors/year, likely Condon/Narayanan-adjacent group, early 1990s]}}
@inproceedings{multipacket1992, title={Multi-Packet Selection on Mesh-Connected Processor Arrays}, booktitle={IPPS}, year={1992}, note={ADS 1992ipps.conf...37K; [verify authors]}}

% ---- Distributed DB top-k (C2) ----
@article{fagin2003optimal, author={Fagin, Ronald and Lotem, Amnon and Naor, Moni}, title={Optimal Aggregation Algorithms for Middleware}, journal={J. Computer and System Sciences}, volume={66}, number={4}, year={2003}}
@inproceedings{cao2004tput, author={Cao, Pei and Wang, Zhe}, title={Efficient Top-K Query Calculation in Distributed Networks}, booktitle={PODC}, year={2004}}
@inproceedings{michel2005klee, author={Michel, Sebastian and Triantafillou, Peter and Weikum, Gerhard}, title={KLEE: A Framework for Distributed Top-k Query Algorithms}, booktitle={VLDB}, year={2005}, url={https://www.vldb.org/archives/website/2005/program/paper/thu/p637-michel.pdf}}

% ---- FPGA / hardware sorting & selection (C2) ----
@inproceedings{samardzic2020bonsai, author={Samardzic, Nikola and Qiao, Weikang and Aggarwal, Vaibhav and Chang, Mau-Chung Frank and Cong, Jason}, title={Bonsai: High-Performance Adaptive Merge Tree Sorting}, booktitle={ISCA}, year={2020}, doi={10.1109/ISCA45697.2020.00033}}
@article{papaphilippou2021flims, author={Papaphilippou, Philippos and others}, title={FLiMS: a Fast Lightweight 2-way Merger for Sorting}, note={arXiv:2112.05607}}
@inproceedings{qiao2022topsort, title={TopSort: A High-Performance Two-Phase Sorting Accelerator Optimized on HBM-Based FPGAs}, booktitle={FCCM}, year={2022}, note={arXiv:2205.07991; [verify authors]}}
@article{jalilvand2023sorting, author={Jalilvand, Amir Hossein and others}, title={Sorting it out in Hardware: A State-of-the-Art Survey}, note={arXiv:2310.07903}}
@article{heavyhitters2023fpga, title={Finding the Top-K Heavy Hitters in Data Streams: A Reconfigurable Accelerator Based on an FPGA-Optimized Algorithm}, journal={Electronics}, volume={12}, number={11}, pages={2376}, year={2023}, doi={10.3390/electronics12112376}}
@inproceedings{parravicini2021topkspmv, author={Parravicini, Alberto and others}, title={Scaling up HBM Efficiency of Top-K SpMV for Approximate Embedding Similarity on FPGAs}, booktitle={DAC}, year={2021}, note={arXiv:2103.04808}}
@inproceedings{wang2021spatten, author={Wang, Hanrui and Zhang, Zhekai and Han, Song}, title={SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning}, booktitle={HPCA}, year={2021}, note={arXiv:2012.09852}}
@article{focus2025, title={Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models}, note={arXiv:2512.14661; [verify authors — read full text: chunked bitonic running-threshold engine]}}

% ---- Streaming / classic (C3) ----
@inproceedings{charikar2002countsketch, author={Charikar, Moses and Chen, Kevin and Farach-Colton, Martin}, title={Finding Frequent Items in Data Streams}, booktitle={ICALP}, year={2002}}
@inproceedings{metwally2005spacesaving, author={Metwally, Ahmed and Agrawal, Divyakant and El Abbadi, Amr}, title={Efficient Computation of Frequent and Top-k Elements in Data Streams}, booktitle={ICDT}, year={2005}}

% ---- Tenstorrent / Tensix (C4) ----
@inproceedings{brown2024grayskull, author={Brown, Nick and Barton, Ryan}, title={Accelerating Stencils on the Tenstorrent Grayskull RISC-V Accelerator}, booktitle={SC24 Workshops}, year={2024}, note={arXiv:2409.18835}}
@article{matmul2025tenstorrent, title={Assessing Tenstorrent's RISC-V MatMul Acceleration Capabilities}, note={arXiv:2505.06085; [verify authors]}}
@inproceedings{brown2025fft, author={Brown, Nick and Davies, Joseph and Le Clair, Felix}, title={Exploring Fast Fourier Transforms on the Tenstorrent Wormhole}, booktitle={ISC 2025 Workshops}, year={2025}, note={arXiv:2506.15437; [verify given names]}}
@article{stencil2026wormhole, title={Stencil Computations on Tenstorrent Wormhole}, year={2026}, note={arXiv:2605.07599; [verify authors]}}
@article{numkernels2026wormhole, title={Numerical Kernels on a Spatial Accelerator: A Study of Tenstorrent Wormhole}, year={2026}, note={arXiv:2603.23343; [verify authors]}}
@article{fusion2026tensix, title={Operator Fusion for LLM Inference on the Tensix Architecture}, year={2026}, note={arXiv:2606.09879; [verify authors]}}
@inproceedings{amati2025nbody, author={Amati, ... and others}, title={Accelerating Gravitational N-Body Simulations Using the RISC-V-Based Tenstorrent Wormhole}, booktitle={SC'25 Workshops}, year={2025}, note={arXiv:2605.02744 / 2509.19294; [verify]}}
@inproceedings{vasiljevic2024blackhole, author={Vasiljevic, Jasmina and Capalija, Davor}, title={Blackhole \& TT-Metalium: The Standalone AI Computer and Its Programming Model}, booktitle={Hot Chips 36}, year={2024}}
@techreport{blackholemicrobench2025, title={Dissecting the Tenstorrent Blackhole Architecture via Microbenchmarking}, year={2025}, note={Tech report formerly at https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf (404 as of 2026-08-16); cited as ref [8] of arXiv:2605.07599. Not peer-reviewed ASPLOS despite the domain. [locate archived copy before submission]}}
@techreport{jia2019ipu, author={Jia, Zhe and Tillman, Blake and Maggioni, Marco and Scarpazza, Daniele Paolo}, title={Dissecting the Graphcore IPU Architecture via Microbenchmarking}, institution={Citadel}, year={2019}, note={arXiv:1912.03413}}
@misc{poplar_topk, title={TopK --- Poplar and PopLibs API Reference (popops)}, howpublished={\url{https://docs.graphcore.ai/projects/poplar-api/en/3.1.0/poplibs/popops/TopK.html}}}

% ---- Float ordering software prior art (C4 framing) ----
@misc{herf2001radix, author={Herf, Michael}, title={Radix Tricks}, year={2001}, howpublished={\url{http://stereopsis.com/radix.html}}}
@article{merrill2011radix, author={Merrill, Duane and Grimshaw, Andrew}, title={High Performance and Scalable Radix Sorting}, journal={Parallel Processing Letters}, volume={21}, number={2}, year={2011}}

% ---- Other many-cores (C1 context) ----
@article{zaruba2021manticore, author={Zaruba, Florian and Schuiki, Fabian and Benini, Luca}, title={Manticore: A 4096-Core RISC-V Chiplet Architecture for Ultraefficient Floating-Point Computing}, journal={IEEE Micro}, volume={41}, number={2}, year={2021}}
@article{occamy2025, title={Occamy: A 432-Core Dual-Chiplet Dual-HBM2E 768-DP-GFLOP/s RISC-V System for 8-to-64-bit Dense and Sparse Computing in 12nm FinFET}, note={arXiv:2501.07330; [verify authors — PULP/ETH]}}

% ---- Agentic methodology ----
@article{ouyang2025kernelbench, author={Ouyang, Anne and others}, title={KernelBench: Can LLMs Write Efficient GPU Kernels?}, year={2025}, note={arXiv:2502.10517; [verify]}}
@techreport{sakana2025aicuda, author={{Sakana AI}}, title={The AI CUDA Engineer}, year={2025}}
@article{accelopt2025, title={AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization}, note={arXiv:2511.15915}}
@article{kforge2025, title={KForge: Program Synthesis for Diverse AI Hardware Accelerators}, note={arXiv:2511.13274; see also arXiv:2606.02963. [CHECK: does its target list include Tenstorrent?]}}
@article{peak2025, title={PEAK: A Performance Engineering AI-Assistant for GPU Kernels Powered by Natural Language Transformations}, note={arXiv:2512.19018}}
```

## Pre-submission action items

1. **C4 claim language**: replace "first-public silicon characterization" with "first-public characterization of Tensix datapath numeric semantics and sorting-relevant primitive costs"; cite blackholemicrobench2025 + the 7 perf papers as the existing (non-overlapping) characterization corpus.
2. **C2 related-work paragraph**: lead with Krizanc & Narayanan; state explicitly what theory did not do (full top-k set, real NoC costs, measurement). Frame the cost model as the mesh-theory bound instantiated with measured constants.
3. **C3**: add a WarpSelect-vs-GVR-vs-chunk-skip contrast table (signal source, soundness, granularity, verification passes).
4. **Verify before camera-ready**: all entries tagged `[verify]`; locate an archived copy of the Blackhole microbench PDF (Wayback Machine); read Focus (2512.14661) full text; check KForge's accelerator target list for Tenstorrent.
5. **Useful absence to state**: no sorting/selection publication exists for Cerebras WSE, Groq LPU, SambaNova RDU, Occamy, or Esperanto — the paper fills a genuine gap for the whole dataflow-mesh class.
