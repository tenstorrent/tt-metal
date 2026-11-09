.. SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
.. SPDX-License-Identifier: Apache-2.0

Compute-kernel API ordering checker
===================================

The compute kernel API (``tt_metal/hw/inc/api/compute/``) is a thin wrapper over the
stateful LLK API (``tt_metal/hw/ckernels/<arch>/metal/llk_api/``).  Most operations
only work after a matching initialisation call configured the Tensix engine, an init
snapshot must still match the operand/format configuration at execute time, a
different init on the same engine replaces the seated operation, and a curated set
of operations must be uninitialised before the kernel returns.  These rules used to
exist only as prose and were caught (if at all) by the runtime LLK sanitizer or by
wrong results on hardware.

``check_compute_api_order.py`` turns the protocol into data
(``compute_api_order_rules.yaml`` plus the generated
``compute_api_order_sfpu.yaml``) and checks it statically over a kernel's real
compiled translation unit.  It follows calls through the compute API and helper
headers, so hidden init/uninit calls are seen; it analyses each TRISC compilation
separately, so preprocessor and ``if constexpr`` variants are the ones the device
build sees; and it reports a value it cannot prove as *unknown* rather than guessing.

What it reports
---------------

.. list-table::
   :header-rows: 1

   * - Rule
     - Meaning
   * - ``missing-init``
     - an operation has no preceding matching init on some path
   * - ``init-arg-mismatch``
     - the init's compile-time arguments differ from the operation's
   * - ``wrong-init``
     - an init for a different operation (or a reconfiguration that invalidated the
       init) sits between the init and its operation, or a sequence postcondition
       between two inits does not hold
   * - ``missing-uninit``
     - a curated operation that arms persistent state is never uninitialised
   * - ``deprecated-init``
     - a deprecated init API is used (``[[deprecated]]`` or
       ``.github/deprecations.json``)
   * - ``deprecated-api``
     - any other deprecated API is used (for example the legacy
       ``acquire_dst`` / ``release_dst`` pair)
   * - ``loop-invariant-init``
     - warning only: an init that does not depend on the loop variable is re-run
       every iteration

Every finding names the rule, the kernel source line, and the earlier call it
conflicts with.  The checker is advisory: it exits 0 for findings unless
``--fail-on-findings`` is passed.  A run that cannot analyse an input exits 2.

Deprecation is read from the called function's C++ declaration, so a new
``[[deprecated]]`` annotation is reported without editing the rule table.
``deprecated-init`` covers init-style functions; every other deprecated function
is ``deprecated-api``.

Sequence postconditions live in the ``sequence_rules`` table.  After following a
trigger init body, the checker verifies that a required operation was seated with
matching arguments.  For example, ``tilize_init(icb, block, ocb)`` must re-point
the packer at ``ocb``; otherwise a previous operation's tile geometry is silently
reused (issue #52175).  A reconfiguration that invalidates a seated init is the
same mechanism in the other direction (issue #61146).

Helper-local ``constexpr`` predicates are folded from the function body when all
of their template arguments are compile-time known.  This resolves paths such as
``constexpr bool is_sfpu = is_sfpu_reduce_path<...>()`` in the compute kernel
library instead of assuming the unknown branch.



Running it locally
------------------

The checker needs the Clang Python bindings (``libclang`` >= 17) and the SFPI
headers.  The tt-metal build installs SFPI under ``runtime/sfpi``.

.. code-block:: bash

   pip install clang pyyaml

   # One kernel, one thread.  The checker synthesizes JIT-like flags.
   python3 tt_metal/tt-llk/infra/check_compute_api_order.py \
       --kernel path/to/compute.cpp --arch wormhole_b0 --thread MATH

   # A whole compile database (the kernel clang-tidy flow can produce one):
   python3 tt_metal/tt-llk/infra/check_compute_api_order.py \
       --compile-db /tmp/kernel_tidy/compile_commands.json

   # A captured JIT build log (the recommended way to get real per-kernel flags):
   TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_LOGGER_LEVEL=info <workload> 2>&1 | tee build.log
   python3 tt_metal/tt-llk/infra/check_compute_api_order.py --log build.log

Use ``--format json`` for machine-readable output and ``--output FILE`` to write
the report.  ``--list-rules`` prints the operation table with citations.

The corpus runner aggregates findings over discovery or a captured database and
renders a Markdown report with coverage, triage, and false-positive accounting.
It uses a supervised multiprocessing pool: workers reuse one libclang parse
environment, recycle periodically to bound memory, and a worker crash is recorded
as a coverage hole instead of losing the run.  The current discovery set is 758
kernel files across ``ttnn``, ``models``, ``tests`` (including
``test_kernels/``), ``tt-train`` and ``tt_metal``.  The report keeps advisory
warnings (currently ``loop-invariant-init``) in a separate table, so the error
table's total and false-positive rate are not diluted by warning-only findings.

.. code-block:: bash

   python3 tt_metal/tt-llk/infra/run_compute_api_order_corpus.py \
       --discover --arch wormhole_b0 --arch blackhole \
       --threads UNPACK MATH PACK --jobs 8 --max-paths 4 \
       --output /tmp/compute_api_order_corpus.md

   python3 tt_metal/tt-llk/infra/run_compute_api_order_corpus.py \
       --log build.log --triage triage.yaml \
       --output /tmp/compute_api_order_corpus.md

Corpus report
-------------

The reviewed bounded corpus run lives next to the checker:

- ``tt_metal/tt-llk/infra/compute_api_order_corpus.md`` — per-rule counts, each
  finding, its triage verdict, and the false-positive rate;
- ``tt_metal/tt-llk/infra/compute_api_order_corpus_triage.yaml`` — the exact
  fingerprint-to-verdict map used to render the report;
- ``tt_metal/tt-llk/infra/compute_api_order_true_positives.md`` — the reviewed
  true positives grouped by pattern with the suggested fix;
- ``tt_metal/tt-llk/infra/compute_api_order_ttsim_verification.md`` — the
  CPU-only ttsim run, including the injected-bug comparison.

Adding a rule
-------------

1. Add an operation (or transition) to
   ``tt_metal/tt-llk/infra/compute_api_order_rules.yaml``.  Key it on the LLK API
   function that performs the state change, not on a compute-API wrapper: wrappers
   are expanded by the checker and can hide a different init or an uninit.
2. Fill ``match.template`` / ``match.args`` with the ``[init index, execute index]``
   pairs that must agree, and list the fields under ``compared_fields`` so the
   sanitizer-agreement test can check them against the ``SAN_HOOK`` field names.
3. Add the operation to ``invalidated_by`` when a reconfiguration changes a field
   the init snapshotted.
4. Cite the source: an API doc comment, the sanitizer hook, or the bug that
   motivated the rule.
5. Add positive and negative kernels under
   ``tt_metal/tt-llk/infra/tests/compute_api_order_fixtures/`` and a case in
   ``tests/test_check_compute_api_order.py``.

SFPU ``*_tile_init`` / ``*_tile`` pairs are generated from
``ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp``.  To refresh
them run ``python3 tt_metal/tt-llk/infra/gen_compute_api_order_sfpu.py``; the tests
fail if the committed YAML is stale.

The sanitizer-agreement test parses the ``SAN_HOOK`` calls in both architecture
trees and fails if the table and the LLK Sanitizer disagree about an operation they
both cover.  This keeps the static rules and the runtime reference model from
drifting apart.

Limitations
-----------

* A value that is not a compile-time constant is reported as unknown.  The checker
  never guesses a mismatch from a runtime value.
* A kernel translation unit that clang cannot parse (typically because host-generated
  defines or descriptor headers are absent) is reported as a coverage hole in the
  corpus report, never as clean.
* ``--log`` replays the SFPI paths baked into the captured command.  SFPI 7.94's
  analysis-tool fallback has a header-ordering regression that affects clang only;
  pass ``--sfpi`` with an analysis-compatible toolchain (7.73 works) when replaying a
  log produced by a 7.94 device build.  The synthesized ``--kernel`` / ``--discover``
  modes are unaffected.
* Loop analysis is may-analysis: an init inside a loop is assumed not to cover the
  next iteration, and the loop-invariant check is advisory.
* The corpus runner explores a bounded number of correlated condition paths per
  translation unit (``--max-paths``, default 8) and recycles workers after 50
  units by default; both bounds trade some recall for runtime and memory.
* Quasar, dataflow (NoC) kernels, the folded LLK 2.0 execution APIs, and changes to
  the compute/LLK APIs are out of scope, matching the bounty terms.  The SFPU
  init/operation pairs are not modelled by the LLK sanitizer and are checked only
  from the host-side generator.
* SFPU register-assignment order is not modelled.  The #50381 failure was an
  API-internal per-op rewrite of a once-per-kernel SFPU config register, not a
  kernel-level init/execute ordering mistake; the checker reports the latter.

CI
--

``.github/workflows/llk-compute-api-order.yaml`` runs the unit tests and a bounded
corpus sweep as a **non-blocking** check (``continue-on-error: true``) and uploads
the report as an artifact.  The full corpus run uses a captured kernel compile
database from the weekly kernel clang-tidy job.
