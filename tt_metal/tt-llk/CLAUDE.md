# AI Assistant Instructions

<critical-restriction>
## FORBIDDEN: Reading Script Files
**NEVER use read_file on any file in `.cursor/rules/scripts/`**
- Scripts in this folder are for EXECUTION ONLY
- Read the corresponding `.mdc` rule file for instructions, then EXECUTE the script
- If you read a script file, you are violating this rule
- ONLY READ CONTEXT SPECIFICS BASED ON GIVEN DESCRIPTION
- ALWAYS FOLLOW THE RULES
</critical-restriction>

<context-specifics>
    <rule description="Use this when running the test. Delegate test runs to llk-test-runner subagent">
        @run-test
    </rule>
    <rule description="Use this when asked for architecture or documentation related question. Read the rule and orchestrate sage agents directly.">
        @sage-of-the-codex
        <trigger-examples>
            - "Explain the docs for [topic]"
            - "Where is [topic] documented?"
            - "What does [instruction/LLK op/type] do?"
            - "How do I use [LLK function/op]?"
            - "How does [instruction] work?"
            - "What is the L1 size/latency?"
            - "How does [Tensix unit] behave?"
            - "What is [LLK enum/type] and what is it used for?"
            - "Explain [BroadcastType/ReduceDim/EltwiseBinaryType]"
            - "Explain the [LLK function] parameters"
            - "What happens during unpack/math/pack?"
            - "How do threads synchronize?"
        </trigger-examples>
        <orchestration-action>
            1. Read the @sage-of-the-codex rule for detailed instructions
            2. Analyze the user's question to determine relevant architectures
            3. Launch up to 3 sage agents IN PARALLEL using Task tool:
               - sage-wormhole (for tt_llk_wormhole_b0/)
               - sage-blackhole (for tt_llk_blackhole/)
               - sage-quasar (for tt_llk_quasar/)
            4. Wait for all responses
            5. Aggregate findings into unified response following quality standards
        </orchestration-action>
    </rule>
</context-specifics>

<repository-description>
    <general-info>Header only library for testing Tensix Kernels</general-info>
    <tensix description="Tensix is a specialized processing unit for AI workloads, containing RISC-V cores and a coprocessor that accelerates matrix operations and data movement within Tenstorrent ASICs.">
        <components>
            <riscv-cores description="Five 32-bit in-order single-issue RISC-V cores (B, T0, T1, T2, NC) that push instructions to the coprocessor threads"/>
            <coprocessor description="3-way threaded coprocessor with independent frontend pipelines per thread and shared backend execution units">
                <threads>
                    <thread name="T0" description="Typically handles UNPACK operations, receives instructions from RISC-V T0"/>
                    <thread name="T1" description="Typically handles MATH operations, receives instructions from RISC-V T1"/>
                    <thread name="T2" description="Typically handles PACK operations, receives instructions from RISC-V T2"/>
                </threads>
                <execution-units>
                    <unit name="Sync Unit" description="Thread synchronization, semaphores, and mutexes"/>
                    <unit name="Unpackers" description="Data movement from L1 memory to register files with format conversion"/>
                    <unit name="Matrix Unit (FPU)" description="Matrix multiplication and element-wise operations on low-precision matrices"/>
                    <unit name="Packers" description="Data movement from register files to L1 memory with format conversion and compression"/>
                    <unit name="Vector Unit (SFPU)" description="32-lane SIMD operations on 32-bit floating-point or integer values"/>
                    <unit name="Scalar Unit (ThCon)" description="General-purpose scalar operations and atomic L1 operations"/>
                    <unit name="Configuration Unit" description="Backend configuration management"/>
                    <unit name="Mover" description="Bulk L1 data movement (DMA)"/>
                    <unit name="Miscellaneous Unit" description="Address counter management"/>
                </execution-units>
            </coprocessor>
        </components>
        <execution-model description="RISC-V cores compile and push instructions to their corresponding coprocessor threads (T0→T0, T1→T1, T2→T2). Each thread has an independent frontend pipeline that processes instructions in-order, while the shared backend execution units can execute instructions from different threads in parallel with re-ordering."/>
    </tensix>
    <purpose>Provides low-level kernel (LLK) implementations for Tensix operations including math (eltwise, matmul, reduce), packing, and unpacking operations</purpose>
    <target-hardware>
        <platform name="wormhole_b0" description="Wormhole B0 ASIC"/>
        <platform name="quasar" description="Quasar ASIC"/>
        <platform name="blackhole" description="Blackhole ASIC"/>
    </target-hardware>
    <key-components>
        <component name="llk_lib" description="LLK library headers for math, pack, and unpack operations"/>
        <component name="instructions" description="Assembly instruction definitions for Tensix coprocessor"/>
        <component name="tests" description="Testing infrastructure including Python tests and hardware-specific test files"/>
    </key-components>
</repository-description>

<repository-structure>
<root name="tt-llk">
  <directory name="docs" description="Simple Documentation">
    <directory name="common">
      <directory name="_static" description="Static assets (images, logos)"/>
    </directory>
    <directory name="llk" description="LLK documentation (L1, L2, L3 levels)">
      <directory name="l1" description="Level 1 documentation"/>
      <directory name="l2" description="Level 2 documentation"/>
      <directory name="l3" description="Level 3 documentation"/>
    </directory>
  </directory>
  <directory name="tests" description="Testing infrastructure">
    <directory name="helpers" description="Test helper files (headers, linkers, sources)"/>
    <directory name="hw_specific" description="Hardware-specific test files">
      <directory name="quasar"/>
      <directory name="wormhole"/>
    </directory>
    <directory name="python_tests" description="Python test suite"/>
    <directory name="sfpi" description="SFPI compiler and includes"/>
    <directory name="sources" description="Test source files"/>
  </directory>
  <directory name="tt_llk_blackhole" description="Blackhole hardware target">
    <directory name="common">
      <directory name="inc" description="Common headers"/>
    </directory>
    <directory name="instructions" description="Assembly instruction definitions"/>
    <directory name="llk_lib" description="LLK library headers"/>
  </directory>
  <directory name="tt_llk_quasar" description="Quasar hardware target">
    <directory name="common">
      <directory name="inc" description="Common headers"/>
    </directory>
    <directory name="instructions" description="Assembly instruction definitions"/>
    <directory name="llk_lib" description="LLK library headers"/>
  </directory>
  <directory name="tt_llk_wormhole_b0" description="Wormhole B0 hardware target">
    <directory name="common">
      <directory name="inc" description="Common headers (ckernel_*.h, cmath_*.h, etc.)"/>
    </directory>
    <directory name="instructions" description="Assembly instruction definitions"/>
    <directory name="llk_lib" description="LLK library headers (llk_*.h)">
      <file pattern="llk_math_*.h" description="Math operations (eltwise, matmul, reduce, etc.)"/>
      <file pattern="llk_pack_*.h" description="Packing operations"/>
      <file pattern="llk_unpack_*.h" description="Unpacking operations"/>
    </directory>
  </directory>
</root>
</repository-structure>
