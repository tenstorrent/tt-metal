# SWE5 singleton comparison candidate

Prepared local source-only candidate `35baa4fb1078675ec99341d082c222cc61877e7e` on `bb16e00efd8bbc649d68d21b501acec1d53e42a9` in isolated branch `prep/swe5-b1-singleton`. Only configuration changes are SWE concurrent trials2→1, max_num_seqs8→1 and corresponding capacity metadata8→1. Exact frozen five IDs/order, limits, seeds, resources, native tools/sampling and retention remain unchanged; full configuration source equality after reversing those three scalars is recorded in config-audit.json.

Validation:34 focused tests passed; all repository hooks passed (ruff, formatting, full pytest hook, copyright), diff check clean. Four source/test files committed; working tree clean. No device use, dispatch, push, publication-checkout writes or central edits. Device forecast/actual/box0.

This prepares an ordinary same-five C1/B1 scored comparison only. It is not accepted until compared against a valid current C2 retained rung; live-log validity blocker remains. A single C1 five-task job shares one server setup and cache lifetime, while five singleton partitions each incur independent cold startup, sandbox/network setup, queue and tail costs. Thus this comparison cannot validate partitioned release timing or environment behavior; those require the later actual frozen-route dry run and measured projection.
