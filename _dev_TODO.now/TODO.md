./build_metal.sh -c -e --debug --without-python-bindings

There is an activity to migrate all device operations code to a new TMP-style: https://github.com/tenstorrent/tt-metal/issues/32680
- The timeline is strict, few weeks
- It is required to validate this migration

Possible path forward is:
- Having a "golden-standard" implementation (Fixed subtasks of 32680)
- Having a set of global "rules and best practices" of TMP implementation
(Can be inferred from current problem descriptions with examples)

- To identify a per-OP set of invariants / "rules" which must be obeyed for the implementation to be marked as "correct"
- To perform checks, starting with LLM-only approach (then C++ parsing tools should be also used to perform validations)

After an initial investigation, approach and details will be discussed/aligned with Artem Yerofieiev.


- !!! To verify that per-OP mds correspond to he refernce implementation
- to cleanup C++ code validator PoC from a branch

- !!! To verify consistency of all .md


[New LLK-related]
~ To check implementations - tt-metal Tasks:
    + dropout
    + convert_to_chw
    +~ experimental/cnn/convert_to_hwc/device/convert_to_hwc_op
        ~ weird, transpose_wh_init_short(cb_in); <-- _short version is used, differs from to_chw!
-            To check state machine transitions, when this is correct, which is preferrable


    "Migrate ops to a single latest infra"
        https://github.com/tenstorrent/tt-metal/issues/32680
        "Whole list of tasks"
            "to tmp" https://github.com/tenstorrent/tt-metal/pull/11793
            "TMP Infra" https://github.com/tenstorrent/tt-metal/pull/11956
            "experimental/" https://github.com/tenstorrent/tt-metal/tree/main/ttnn/cpp/ttnn/operations/experimental/dropout


~!! to improve LLK entities content
    to have the overall glossary of Tetirs-blocks, inputs, outputs, state machine descriptions
            can be used to decide which LLKs will perform the required data flow transformation
        written both as a list
        AND a mermaid diagram
    to have per-LLK-entity files describing constraints, usual data transformations applied [expec uint32, but usually passed prob as float 0..1]

    - TODO: in optimized step, to insert loop unrolling with hints why
        based on convert_to_chw - there is loop unrolling via templated function pattern with rest processing one by one
            this pattern can be formulated

- looks like a bug, but it is NOT, it is a part of LLK contract
    uint32_t uscale = std::bit_cast<uint32_t>(args.scale);
        -> static_cast should be used or so?
            scale_factor    | uint bitwise representation of 32 bit floating point scale factor          | uint32_t |


- To make a plan how to validate all these
-! ASK:
    get started by looking at exising tt-metal repo
        building up a couple sets of data we can use for evaluation and training

    - Self-written with domain knowledge of C++ via c++flirt or whatever, etc libs in future?

            2 main things:
                Now it complicated to write new kernels
                    in tt-metral you need to write program factory
                    what reader kernel, what computer kernel, buffers
                        then make this

            We'll create some manually
                hopefully define some workflow - some AI system
                    we can go and extract
                        or if not feasulbe - manuyll

            2) Kernel debug side
                existing Kernel - works, then break kernel
                can be good evaluatie
                    to see broken kernel and try to fix

    - Which are groups of OPs?
        - Some of them are pure using some HW op
        - Some are complex

    - [tt-llk] LLK specifications might be hard to express with all limitations, etc

    ~ [tt-metal] Simpler task might be + will bring value right now
        is to formulate and validate currently transitioned to tmp OP-related code


Which are limitations:
- [Global] TMP style of the code
    Can be solved by Description with example, already exist

- "invariants" which should be met
    - [per - OP] Should be
        - To generate these from existing golden TMP implementations
            - To formalize them with an ability to 1-to-1 map C++ implementation to the description
