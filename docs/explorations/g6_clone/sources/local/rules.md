## Model environment

The op runs inside a model. For every op of this campaign the following holds; the model-wide rules
under `## Rules` state it as obligations:
- one call of the op is one program in the model's sequence of programs;
- every tensor of the signature is the model's own tensor, in the shape, dtype, layout, memory
  placement and packing named there: no program runs between the op that produces a tensor and this
  one;
- another device program starts right after every call;
- the model keeps tensors of its own in L1 while the op runs, and leaves 1347776 bytes per core to
  the op;
- the op's neighbours in the model are generated separately against the same interface contract, so
  the boundary of this op does not move;
- the model replays its calls from captured traces;

;; The section above goes before `## Rules`; the lines below go LAST under `## Rules`, unchanged.

Model-wide rules (the same text in every op spec of this campaign; ids are those of the black
bible):

- When the entry point is called (every call): the call MUST run as one device program -- MUST NOT
  dispatch a second program, chain other device ops or read tensor values on the host; the fixed
  cost of a program and every extra crossing of DRAM is a cost that no kernel tuning removes, the
  values stay right and no golden cell counts programs (BB-01).
- When the entry point runs (every call): the op MUST take and return every tensor in the shape,
  dtype, layout, memory placement and packing this spec names for it -- the entry point MUST NOT
  call a slice, reshape, typecast, layout or memory conversion on an input or an output, and the
  caller MUST NOT need one before or after the call; such programs scale with users and tokens and
  cancel the op's gain, and the values of a wrapped call are right (BB-02).
- When another program runs right after the call (always in the model): every kernel MUST have
  completed every NoC transaction it issued, remote semaphore increments and other atomic operations
  included, when it returns -- a transaction left in flight hangs the next program, and a single
  call or a profiled run hides it (BB-03).
- When the kernels of the op are built (every build flavour the campaign runs: plain, with profiler
  zones, with NoC tracing): the kernels of one core with their runtime arguments and buffer
  configuration MUST fit the kernel-config region of that core -- MUST NOT fit only in the leanest
  build; an image that is too large is a refused launch that no golden cell of another build shows,
  and an op that cannot be traced cannot be diagnosed (BB-04).
- When the model holds tensors of its own in L1 during the call (every call of the model): the L1
  the op uses on any core MUST stay within 1347776 bytes for every shape of the golden INPUTS, and a
  shape that cannot meet the bound MUST be refused on the host -- MUST NOT pass only when the op has
  a whole L1 to itself; a test that calls the op alone hides the overlap, which then fails at launch
  only for the shapes that reach it (BB-05).
- When the same call is repeated on the same inputs and state (every shape): the op MUST return
  bit-identical outputs and state on every run -- a hand-over that is taken before its data has
  landed gives a result that changes between runs and still passes a tolerance most of the time
  (BB-06).
- When the op is called again with the same shapes, dtypes, layouts, memory configurations and
  options (every repeat call): the host MUST NOT rebuild the program description, and everything
  that changes the program MUST be part of the cache key -- an untraced call otherwise pays the
  build on every call, and an incomplete key replays a wrong program (BB-07).
- When the op chooses among internal plans by shape or by free L1 (every call of such an op): the op
  MUST be able to report the plan a call selected, and a production shape MUST get the same plan
  with its inputs placed as the model places them -- a silent fall to a slower plan passes every
  correctness test (BB-08).
- When the op is refined or tuned (every later change): what the op takes, computes and returns MUST
  stay what this spec says -- MUST NOT move work across the boundary to or from a neighbouring op to
  gain time, because the neighbours are generated separately against the same contract; inside the
  boundary every restructuring is allowed (BB-09).
- When the op carries fp32 state across calls or steps (every call of such an op): the carried state
  and every full-precision factor applied to it MUST NOT pass through a matrix-engine operand of
  reduced precision, and the state MUST stay fp32 at every call boundary -- a state that is cut at
  every step drifts without bound and still passes a loose limit (BB-12).
- When fp32 values that feed a carried state are computed with matrix-engine products (every such
  product): the one-sided truncation of the operands MUST be compensated so that the state error
  against the fp32 reference, whole tensor and worst unit, does not grow on the state-carry cases of
  the suite -- truncation accumulates in the state and a correlation check does not see it (BB-13).
- When the op runs under trace replay (every call of a traced model): a call after capture MUST NOT
  allocate a device tensor or build a kernel, and a replayed call MUST equal the eager call bit for
  bit -- a late allocation or a build at request time stalls or breaks the replay (BB-16).
- When a kernel group of the op spans more than one core (every multi-core shape): for every
  production shape the cores of each kernel group MUST form few rectangular core ranges, reached by
  choosing which core gets which role and not by changing the data path -- binaries and launch
  messages are written once per core range, so scattered groups start late on every call of a traced
  model (BB-17).
- When the op runs under trace replay (every call of a traced model): the kernel text of every role
  MUST be small enough to sit in the kernel-config region beside the binaries of the programs that
  precede the op -- MUST NOT be reached with a size-optimising build that slows the kernels;
  otherwise the dispatcher waits for the previous program to end before it writes the op's binaries
  and launch messages (BB-18).
- When the op is built for profiling (the zone build): every piece of work and every blocking wait
  of every core MUST fall inside its own named zone, the op MUST be able to dump which role each
  core has, and none of this MUST change the plain build -- time outside a zone reads as waiting,
  and the wrong role gets optimised (BB-21).
- When tests of the op are written or run (every phase): the tests MUST be limited to the production
  cases this spec names and to the cases its rules require, and a run MUST cover only the cases the
  change at hand can affect -- sweeps over shapes, dtypes or layouts that production never calls and
  full-suite runs after every edit take most of the device time and decide nothing (BB-22).
