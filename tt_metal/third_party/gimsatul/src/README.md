# Gimsatul SAT Solver

This is a portfolio style parallel SAT-solver which physically shares
clauses between different solving threads.  This is made possible by using
separate watcher data structures for each solver thread and keeping the
actual clause data immutable.  This allows to share large clauses between
threads through atomic reference counting, which in turn allows not only
sharing the original large (and binary) clauses but also much more
aggressive sharing of learned clauses during the search while keeping the
overall memory foot-print small.

## References

More details can be found in the first report on Gimsatul submitted and presented
at
[(POS'22)](http://www.pragmaticsofsat.org/2022):


<a href="https://cca.informatik.uni-freiburg.de/fleury/index.html#publications">Mathias Fleury</a>
and
<a href="https://cca.informatik.uni-freiburg.de/biere/index.html#publications">Armin Biere</a>.
<br>
<a href="https://cca.informatik.uni-freiburg.de/papers/FleuryBiere-ARXIV22.pdf">Scalable Proof Producing Multi-Threaded SAT Solving with Gimsatul through Sharing instead of Copying Clauses</a>.
CoRR abs/2207.13577 (2022), presented at
<a href="http://www.pragmaticsofsat.org/2022/">12th Workshop on Pragmatics of SAT (POS'22)</a>.
<br>
[ <a href="https://cca.informatik.uni-freiburg.de/papers/FleuryBiere-ARXIV22.pdf">paper</a>
| <a href="http://arxiv.org/abs/2207.13577">arxiv</a>
| <a href="https://cca.informatik.uni-freiburg.de/papers/FleuryBiere-ARXIV22.bib">bibtex</a>
| <a href="https://github.com/arminbiere/gimsatul">gimsatul</a>
| <a href="https://cca.informatik.uni-freiburg.de/biere/talks/Biere-POS22-talk.pdf">slides</a>
]

You might also want to check out our SAT competition proceedings entries for Gimsatul which can be found at our 
<a href="https://cca.informatik.uni-freiburg.de/papers">publication</a>
page.

## Sharing

Interesting learned clauses are exported (shared) rather aggressively and
imported eagerly before making a decision.  Beside exchanging all learned
units between threads, there is a preference given to import low-glue
learned clauses (glue = LBD = glucose level), with binary clauses having
highest priority, followed by glucose level one clauses, then tier 1 clauses
(glue of two) and finally tier 2 clauses (glue at most 6).  Tier 3 clauses
(with glue larger than 6) are not shared.

Binary original clauses (after preprocessing) are not allocated but kept
virtual in separate watcher stacks, which then are shared among all threads
(as they are not changed).  Learned binary clauses are virtual too but
kept in thread local watcher lists, and thus are the only part really
physically copied.

## Proofs

From a memory as well as proof perspective these shared large clauses occur
only once and are only deleted when their atomic reference count reaches
zero.  Without physically sharing learned clauses, as in most other parallel
solvers, the copied clauses also have to be duplicated in the proof trace
(using multi-set semantics).  Thus physically sharing clauses allows to
produce more compact global DRAT proofs.  Checking those proofs is still not
trivial trough, at least with a sequential proof checker, as proof lines are
produced at a much higher rate than in a sequential solver.

## Preprocessing

As far preprocessing and inprocessing is concerned equivalent literal
substitution, subsumption including strengthening and of course bounded
variable elimination are first run before solvers are cloned to share
original irredundant clauses.  The same code is used to simplify the
formula in regular intervals. This requires all solvers to synchronize.
Then one thread runs the global single-threaded simplification code.
Further inprocessing is scheduled in form of failed literal probing and
vivification locally within each solver thread.  It would also be useful to
parallelize preprocessing, which currently is only run in a single thread
initially before cloning and starting the solver threads.

## Naming

The name of the solver comes from *gimbatul* which is in the "Black Speech"
language invented by Tolkien and occurs in the inscription of the "One Ring"
in "Lord of the Rings" and literally translates to "find them" (all).

We follow that terminology in the source code and the main thread which
performs preprocessing and organizes everything is called the "ruler" and
an actual solver thread is called "ring".

## Build

Use the following to configure, compile and test the solver:

> `./configure && make test`

For more information on build options try:

> `./configure -h`

## Usage

The resulting solver `gimsatul` is multi-threaded but you currently
need to specify the number of threads explicitly to make use of that
feature:

> `./gimsatul cnf/prime4294967297.cnf --threads=16`

Otherwise the number of threads defaults to one thread.  Information about
other command line options can be obtained with

> `./gimsatul -h`

and is also available in [usage.h](usage.h).

The solver reads (optionally compressed) files in DIMACS format
and if requested is able to produce DRUP/DRAT proofs. To generate a
proof trace just specify the path to the output proof file
as an additional argument on the command line

> `./gimsatul cnf/prime4294967297.cnf --threads=16 /tmp/proof`
