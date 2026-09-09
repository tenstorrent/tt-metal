# Version 1.1.3
---------------

- Also eventually reducing tier1 clauses.

- Dynamic tier glue limit computation.

- Glue promotion.

# Version 1.1.2
---------------

- Fixed GITID issue when building from tar balls.

# Version 1.1.0
---------------

- Chronological backtracking (also helps importing units).

- Binary reason jumping.

- Added '--force-phase' option to force every worker ring to only
  use the initial phase as specified with '--phase=[01]' for all
  decisions (even in the default portfolio mode).  Added statistics
  gathering and printing code for phase decisions too.

- Minor compilation issue.

# Version 1.0.3
---------------

As version 1.0.2 in single threaded mode was still lagging behind 'Kissat
SC2021 Light' even after disabling all features that are not in Gimsatul
yet, we found and fixed some issues with implementation of heuristics
and made particularly scheduling of regularly running functions similar to
how they are scheduled in Kissat.

There are still fundamental differences though between this version and
Kissat (watchers, initial preprocessing) but this version running in single
threaded is coming close to the performance of 'Kissat SC2022 Light' (if all
features not in Gimsatul yet are kept disabled).

- Scheduling with an 'O(nlog^4n)' conflict interval for mode switching
  where 'n' is the number of times it was scheduled, 'O(nlog^3n)' for
  rephasing, 'O(nlog^2n)' for simplification, 'O(nlogn)' for probing,
  and 'O (logn)' for restarts, following Kissat in this regard.

- Variables get the same order as in Kissat for the pairing heap (EVSIDS)
  with variables with larger index initially having higher scores.

- Options closer to those used in Kissat (restart intervals, vivify effort,
  simplify interval, switch interval).

- Scaling conflict intervals of probing and simplification as in Kissat with
  respect to the number 'n' of irredundant clauses in 'O (log^2 n)'.

- More verbose signal catching failure messages.

- Counting occurrences during variable elimination incurs 'ticks'.

- Fixed iterating logic (thus more precise 'i' report lines) also
  while importing units.

- Specialized pairing heap slightly.

- More precise importing function needed for importing units
  and clauses during 'vivification'.

- Fixed logic which allows to disable preprocessing or inprocessing.

- Separated vivification for 'tier1' and 'tier2' clauses.

- Sorting of literals in vivified clauses and vivification candidate clauses
  as in Kissat and CaDiCaL to simulate trie based vivification which allows
  allows to reuse some decisions and propagations.

# Version 1.0.2
---------------

This is the first version with most optimizations originally planned but
did not make it into the 'sc2022' version.

- Optimized vivification by first waiting for the next reduce to avoid
  spending time on garbage collected 'tier2' clauses, sorting probed
  literals in the vivified clause by score / stamp and reusing decisions.

- Simplified and improved 'reduce':  reduce only goes over 'tier2' clauses
  ignoring references before the cached start of the first 'tier2' clause
  if no new fixed root-level unit was found; flushing and mapping references
  was merged in order to have just one pass over the watchers.

- Removed the compile-time option './configure --no-middle'.

- The pools for sharing clauses are now indexed by the glue of the shared
  clause which makes it more fine grained.  We now also allow clauses to
  be shared with much larger glue (at least up-to glue 15 as this makes the
  pools 128 bytes big, thus the size of a cache-line).  This in essence
  gives a priority queue for importing clauses with lower glue clauses being
  preferred (even if they are older) and for the same glue the newer clause.

- Added './configure --metrics' compile-time option to produce and print
  slightly more expensive statistics (clause visits counters broken down
  by size, much more learned, exported, and imported counters broken down
  by glue and a new shared clause counter with export per shared rate).

- We further increased the watcher stack size by 8 bytes (from 24 to 32)
  making room for storing the literals of clauses of size 3 and 4 directly
  in the watcher.  Thus this avoids another memory dereference to such short
  but non-binary clauses.  For those short clauses the middle reference is
  not needed giving room for one more literal and the previously unused
  padding needed for the 8-byte aligned clause pointer in a watcher gave
  another one.  This explains why increasing the size by 8 bytes gives room
  for 16 bytes (one could in principle stuff in two more literals extending
  the XOR trick, and actually get away with 24 bytes for up to 4 literals,
  but this would make propagation and analysis code much more complex).

- Separated watchers from opaque watch pointers and allocated these watchers
  on thread local watcher stack instead of using allocating thread local
  objects.  This requires indexing (restricts the number of watchers to
  2^31-1) instead of using pointers in watch lists, but makes room for
  blocking literals which seem to speed up propagation substantially also
  for us in this solver (and was the whole point of this change).

- Added termination checks to simplifiers and local search.

- Improved scheduling of simplification limits and bounds.

- Added `-i` / `--id` command line option (to get GIT id even for
  `./configure --quiet`).

- Made the pool cache-line-size aligned to maximizes cache usage and still
  avoid false sharing (as one pool is cache-line sized).

- Removed potential race during logging dereferenced clause.

- Factored out empty clause tracing into `set_inconsistent`
  (as tracing was missing at two calls to it).

- Removed duplicated trace deletion in `compact_ring` during removal of
  root-level falsified literals in saved thread-local learned clauses, when
  the shrunken clause becomes binary (thus virtual).

# Version 1.0.1
---------------

This is in essence the same as the original 'sc2022' version after fixing
the issues found by MacOS users which were not able to compile it.

- Xcode needs `_Atomic(...) *` instead of `volatile ... *` first arguments in
  `atomic_exchange` etc.

- Added `./configure --pedantic` to test for C11 conformance
  (removes `popen` and thus compressed reading).

# Version sc2022
----------------

The version submitted to the SAT Competition 2022 with experiments
described in our POS'22 paper and the SAT Competition system description.
