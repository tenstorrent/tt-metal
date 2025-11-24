My goal is to have [after some more iterations of describing preconditions and invariants for more OPs, analyzing correct implementatins based on which these can be inferred] a clear, precise, no-yapping structured descriptions of OPs which need to be implemented (currently, info about dropout and convert_to_chw were analyzed).

The goal then is to
- Have a clear precise, human-readable and machine-readable, no-yapping specifications of OPs
- Have similar in clarity specifications related to all LLKs nearby
- Have a similar clear description of C++ level API expected and code style of the code related to OPs
==
Be able to
- Re-generate the whole implementation for the OP and compare with git diff
[If something was changed, specifications of LLKs, more options became available, architecture had changed, no golden imlementation exists]
- Create tests to test the implementation in a form of C++ itself, checking that C++ structure adheres to the commmon rules, checking that HW-related implementation uses currect number of CircularBuffers, checking that arguments to Kernels are passing the required data AND Kernels expect these arguments in this order, etc.
- This is many-component system, so having those Base-level specifications and rules, the ability to verify that the implementation corresponds to what was meant to be implemented, everything is written in the same parrern, easy to analyze/extend/plan

At some point I can imagine that these sepcifications are inferred from some larger Tenstorrent architectural more_base document, and all implementations would be correctly implemented and verified on C++ - parsing level, verifying the correctness. The only thing that is happening - there is preparation of the data for LLK functions, which have their own preconditions


====

Please review and improve. The aim to fully formulate all preconditions and invariants, in a simple language which could be [at later project stages] reformulated and simplified more to be able to be put the data from these .md in graph DB.


Make a plan to rewrite/improve .md files in _dev_TODO.now (with per-OP/LLK subfolders).

The idea is to describe the algorithm layered. General computation/data flow [no HW limitations]; Data flow modifications related to HW limitations [blocks, sharding, etc related], without much low-level details.

Then, there are preconditions which must be obeyed when using target ttnn's LLK OPs, this should be clear.

Then, there are limitations on how this algorithm should be written in C++ - more low-level details like accurate Circular Buffers creation, passing their IDs correctly as <THESE> arguments to <THESE> kernels, etc.

=====

Make sure that for an abstract LLM-based code generator, based ONLY THE DATA ABOVE FROM _dev_TODO.now folders and fubfolders will be obvious how to implement OPs fully 1-to-1 matching the implementation below:

========




=========



My task is to prepare a sourse of truth for the code generator. Based purely on this source, the code generator must replicate the solution in the same approach. For this, I'm analyzing golden-implementations in proper style, using correct APIs, following best practices of the project.

Overall, large refactoring process is happening to translate ttnn implementation of OPs [operations] in the style of Templated Metaprogramming Pattern, described here
@ttnn/cursor/DEVICE_OPERATION_MIGRATION_GUIDE.md

======

I would like to formulate invariants and preconditions required in a form of strict, precise statements with aim to store this as .md data in text comsumable for LLMs, but keeping in mind that at some point all this data might be moved to a graph-database with entities and relations between them, with the ability to abstract details of how the system work with knowledge to generate the code according to the specifications.

====

Please extract all the required info from the DEVICE_OPERATION_MIGRATION_GUIDE.md, to have a global Migration.md with a clear sets of rules each implementation should adhere - From the DataFlow perspective (focusing on data flow transformations required to computate the input data) and Structurally [how the implementation should be split to the files or classes]

Please write those in _dev_TODO.now folder to
Global_DataFlow.md and Global_Structural.md

====

I expect a general rules for all OP implementations there, not focusing on particular Dropout implementation or so
