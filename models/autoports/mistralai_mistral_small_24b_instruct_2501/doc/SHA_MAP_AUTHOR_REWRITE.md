# Commit SHA map — author-name rewrite, 2026-08-17

Five commits on this branch were authored under an incorrect name (`Marko
Vasiljevic`, an incorrect expansion of the `mvasiljevic` git identity). They were
rewritten to `Marina Vasiljevic <mvasiljevic@tenstorrent.com>`, and the branch was
force-pushed.

The rewrite changed **content of nothing** — the resulting tree is byte-identical
to the pre-rewrite tree, and the import merge keeps both parents (`d58cb341c70`
and `734625b8744`). Only commit metadata changed.

Because git renumbers every descendant of a rewritten commit, the SHAs recorded
*inside* the stage documents in this tree now refer to commits that no longer
exist. The stage docs were deliberately left as written — they are the agents'
own records — so use this table to resolve them.

| SHA cited in the docs | current SHA | commit |
|---|---|---|
| `37f95c9e2a1` | `1ddac9361dc` | Start Mistral-Small-24B-Instruct-2501 bringup on fast-models-fast |
| `d0313954b04` | `0855a907d49` | Optimize Mistral Small 24B full-model token-out path |
| `d182c2fe795` | `5bd55dd574b` | Keep optimized-full-model CSV evidence local, not in-tree |
| `c86a4a95d1a` | `46b78b59ff8` | Add Mistral Small 24B datatype sweep |
| `3381c24c309` | `510e1b33583` | Integrate Mistral Small 24B with vLLM |
| `1d1d40ffe15` | `ef299119f07` | Keep the raw vLLM server log local, not in-tree |
| `7e14ba84874` | `2b7620048b3` | Record optimized Mistral vLLM serving evidence |
| `cf1410271de` | `1a8eef7bdf2` | Log optimized vLLM checkpoint SHAs |
| `f1d5047c9c6` | `c38b224dd48` | Complete optimized vLLM commit record |
| `cc30b7a1cda` | `6ec509f21b7` | Record optimized vLLM repository branches |
| `1529e332a1c` | `0eb0a8e4b27` | Refresh optimized vLLM artifact provenance |
| `5bab286dc7f` | `4c93bb27f09` | Document Mistral release warning autofix evidence |

Notes:

- Stage 08 records its measurement base commit as `d182c2fe795`; read that as
  `5bd55dd574b`.
- Commits from the pre-merge model lineage
  (`mvasiljevic/model/mistralai-mistral-small-24b-instruct-2501`) are ancestors of
  the import merge and are **unchanged**, so citations such as `29dd518771f`,
  `3d35e46c5b5`, `6154b41e1f8`, `79e40abfccf`, `90994c530de`, `a0f6df651c8`,
  `92f5a3cbf5c` and `64608f66cd8` still resolve directly.
- The two operational fixes stage 11 cites, `971ee6cfcdd` and `aab6d846caf`, live
  in the vLLM submodule/plugin repo rather than this branch, so they are
  unaffected.
