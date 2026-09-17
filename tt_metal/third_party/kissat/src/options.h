#ifndef _options_h_INLCUDED
#define _options_h_INLCUDED

#include <assert.h>
#include <stdbool.h>

#define OPTIONS \
OPTION( ands, 1, 0, 1, "extract and eliminate and gates", 0) \
OPTION( autarky, 1, 0, 1, "enable autarky reasoning", 0) \
OPTION( autarkydelay, 0, 0, 1, "delay autarky reasoning", 0) \
OPTION( backbone, 1, 0, 2, "binary clause backbone (2=eager)", 0) \
OPTION( backbonedelay, 0, 0, 1, "delay backbone computation", 0) \
OPTION( backboneeffort, 20, 0, 1e5, "effort in per mille", 0) \
OPTION( backbonefocus, 0, 0, 1, "focus on not-yet-tried literals", 0) \
OPTION( backbonekeep, 1, 0, 1, "keep backbone candidates", 0) \
OPTION( backbonemaxrounds, 1e3, 1, INT_MAX, "maximum backbone rounds", 0) \
OPTION( backbonerounds, 100, 1, INT_MAX, "backbone rounds limit", 0) \
OPTION( backward, 1, 0, 1, "backward subsumption in BVE", 0) \
OPTION( bumpreasons, 1, 0, 1, "bump reason side literals too", 0) \
OPTION( bumpreasonslimit, 10, 1, INT_MAX, "relative reason literals limit", 0) \
OPTION( bumpreasonsrate, 10, 1, INT_MAX, "decision rate limit", 0) \
OPTION( cachesample, 1, 0, 1, "sample cached assignments", 0) \
DBGOPT( check, 2, 0, 2, "check model (1) and derived clauses (2)", 1) \
OPTION( chrono, 1, 0, 1, "allow chronological backtracking", 0) \
OPTION( chronolevels, 100, 0, INT_MAX, "maximum jumped over levels", 0) \
OPTION( compact, 1, 0, 1, "enable compacting garbage collection", 0) \
OPTION( compactlim, 10, 0, 100, "compact inactive limit (in percent)", 0) \
OPTION( decay, 50, 1, 200, "per mille scores decay", 0) \
OPTION( definitioncores, 2, 1, 100, "how many cores", 0) \
OPTION( definitions, 1, 0, 1, "extract general definitions", 0) \
OPTION( defraglim, 75, 50, 100, "usable defragmentation limit in percent", 0) \
OPTION( defragsize, 1<<18, 10, INT_MAX, "size defragmentation limit", 0) \
OPTION( delay, 2, 0, 10, "maximum delay (autarky, failed, ...)", 0) \
OPTION( eagersubsume, 20, 0, 100, "eagerly subsume recently learned clauses", 0) \
OPTION( eliminate, 1, 0, 1, "bounded variable elimination (BVE)", 0) \
OPTION( eliminatebound, 16 ,0 , 1<<13, "maximum elimination bound", 0) \
OPTION( eliminateclslim, 100, 1, INT_MAX, "elimination clause size limit", 0) \
OPTION( eliminatedelay, 0, 0, 1, "delay variable elimination", 0) \
OPTION( eliminateeffort, 100, 0, 2e3, "effort in per mille", 0) \
OPTION( eliminateheap, 0, 0, 1, "use heap to schedule elimination", 0) \
OPTION( eliminateinit, 500, 0, INT_MAX, "initial elimination interval", 0) \
OPTION( eliminateint, 500, 10, INT_MAX, "base elimination interval", 0) \
OPTION( eliminatekeep, 1, 0, 1, "keep elimination candidates", 0) \
OPTION( eliminateocclim, 1e3, 0, INT_MAX, "elimination occurrence limit", 0) \
OPTION( eliminaterounds, 2, 1, 1000, "elimination rounds limit", 0) \
OPTION( emafast, 33, 10, 1e6, "fast exponential moving average window", 0) \
OPTION( emaslow, 1e5, 100, 1e6, "slow exponential moving average window", 0) \
EMBOPT( embedded, 1, 0, 1, "parse and apply embedded options", 0) \
OPTION( equivalences, 1, 0, 1, "extract and eliminate equivalence gates", 0) \
OPTION( extract, 1, 0, 1, "extract gates in variable elimination", 0) \
OPTION( failed, 1, 0, 1, "failed literal probing", 0) \
OPTION( failedcont, 90, 0, 100, "continue if many failed (in percent)", 0) \
OPTION( faileddelay, 1, 0, 1, "delay failed literal probing", 0) \
OPTION( failedeffort, 50, 0, 1e8, "effort in per mille", 0) \
OPTION( failedrounds, 2, 1, 100, "failed literal probing rounds", 0) \
OPTION( forcephase, 0, 0, 1, "force initial phase", 0) \
OPTION( forward, 1, 0, 1, "forward subsumption in BVE", 0) \
OPTION( forwardeffort, 100, 0, 1e6, "effort in per mille", 0) \
OPTION( hyper, 1, 0, 1, "on-the-fly hyper binary resolution", 0) \
OPTION( ifthenelse, 1, 0, 1, "extract and eliminate if-then-else gates", 0) \
OPTION( incremental, 1, 0, 1, "enable incremental solving", 1) \
LOGOPT( log, 0, 0, 5, "logging level (1=on,2=more,3=check,4/5=mem)", 0) \
OPTION( mineffort, 1e4, 0, INT_MAX, "minimum absolute effort", 0) \
OPTION( minimize, 1, 0, 1, "learned clause minimization", 0) \
OPTION( minimizedepth, 1e3, 1, 1e6, "minimization depth", 0) \
OPTION( minimizeticks, 1, 0, 1, "count ticks in minimize and shrink", 0) \
OPTION( modeconflicts, 1e3, 10, 1e8, "initial focused conflicts limit", 0) \
OPTION( modeticks, 1e8, 1e3, INT_MAX, "initial focused ticks limit", 0) \
OPTION( otfs, 1, 0, 1, "on-the-fly strengthening", 0) \
OPTION( phase, 1, 0, 1, "initial decision phase", 0) \
OPTION( phasesaving, 1, 0, 1, "enable phase saving", 0) \
OPTION( probe, 1, 0, 1, "enable probing", 0) \
OPTION( probedelay, 0, 0, 1, "delay probing", 0) \
OPTION( probeinit, 100, 0, INT_MAX, "initial probing interval", 0) \
OPTION( probeint, 100, 2, INT_MAX, "probing interval", 0) \
NQTOPT( profile, 2, 0, 4, "profile level", 0) \
OPTION( promote, 1, 0, 1, "promote clauses", 0) \
NQTOPT( quiet, 0, 0, 1, "disable all messages", 0) \
OPTION( really, 1, 0, 1, "delay preprocessing after scheduling", 0) \
OPTION( reap, 0, 0, 1, "enable radix-heap for shrinking", 0) \
OPTION( reduce, 1, 0, 1, "learned clause reduction", 0) \
OPTION( reducefraction, 75, 10, 100, "reduce fraction in percent", 0) \
OPTION( reduceinit, 3e2, 2, 1e5, "initial reduce interval", 0) \
OPTION( reduceint, 1e3, 2, 1e5, "base reduce interval", 0) \
OPTION( reducerestart, 0, 0, 2, "restart at reduce (1=stable,2=always)", 0) \
OPTION( reluctant, 1, 0, 1, "stable reluctant doubling restarting", 0) \
OPTION( reluctantint, 1<<10, 2, 1<<15, "reluctant interval", 0) \
OPTION( reluctantlim, 1<<20, 0, 1<<30, "reluctant limit (0=unlimited)", 0) \
OPTION( rephase, 1, 0, 1, "reinitialization of decision phases", 0) \
OPTION( rephasebest, 1, 0, 1, "rephase best phase", 0) \
OPTION( rephaseinit, 1e3, 10, 1e5, "initial rephase interval", 0) \
OPTION( rephaseint, 1e3, 10, 1e5, "base rephase interval", 0) \
OPTION( rephaseinverted, 1, 0, 1, "rephase inverted phase", 0) \
OPTION( rephaseoriginal, 1, 0, 1, "rephase original phase", 0) \
OPTION( rephaseprefix, 1, 0, INT_MAX, "initial 'OI' prefix repetition", 0) \
OPTION( rephasewalking, 1, 0, 1, "rephase walking phase", 0) \
OPTION( restart, 1, 0, 1, "enable restarts", 0) \
OPTION( restartint, RESTARTINT_DEFAULT, 1, 1e4, "base restart interval", 0) \
OPTION( restartmargin, 10, 0, 25, "fast/slow margin in percent", 0) \
OPTION( restartreusetrail, 1, 0, 1, "restarts reuse trail", 0) \
OPTION( seed, 0, 0, INT_MAX, "random seed", 0) \
OPTION( shrink, 3, 0, 3, "learned clauses (1=bin,2=lrg,3=rec)", 0) \
OPTION( shrinkminimize, 1, 0, 1, "minimize during shrinking", 0) \
OPTION( simplify, 1, 0, 1, "enable probing and elimination", 0) \
OPTION( stable, STABLE_DEFAULT, 0, 2, "enable stable search mode", 0) \
NQTOPT( statistics, 0, 0, 1, "print complete statistics", 0) \
OPTION( substitute, 1, 0, 1, "equivalent literal substitution", 0) \
OPTION( substituteeffort, 10, 1, 1e3, "effort in per mille", 0) \
OPTION( substituterounds, 2, 1, 100, "maximum substitution rounds", 0) \
OPTION( subsumeclslim, 1e3, 1, INT_MAX, "subsumption clause size limit", 0) \
OPTION( subsumeocclim, 1e3, 0, INT_MAX, "subsumption occurrence limit", 0) \
OPTION( sweep, 1, 0, 1, "enable SAT sweeping", 0) \
OPTION( sweepclauses, 1000, 0, INT_MAX, "maximum environment clauses", 0) \
OPTION( sweepdepth, 2, 0, INT_MAX, "environment depth", 0) \
OPTION( sweepeffort, 20, 0, 1e4, "effort in per mille", 0) \
OPTION( sweepmaxdepth, 4, 2, INT_MAX, "maximum environment depth", 0) \
OPTION( sweepvars, 100, 0, INT_MAX, "maximum environment variables", 0) \
OPTION( target, TARGET_DEFAULT, 0, 2, "target phases (1=stable,2=focused)", 0) \
OPTION( ternary, 1, 0, 1, "enable hyper ternary resolution", 0) \
OPTION( ternarydelay, 1, 0, 1, "delay hyper ternary resolution", 0) \
OPTION( ternaryeffort, 70, 0, 2e3, "effort in per mille", 0) \
OPTION( ternaryheap, 1, 0, 1, "use heap to schedule ternary resolution", 0) \
OPTION( ternarymaxadd, 20, 0, 1e4, "maximum clauses added in percent", 0) \
OPTION( tier1, 2, 1, 100, "learned clause tier one glue limit", 0) \
OPTION( tier2, 6, 1,1e3, "learned clause tier two glue limit", 0) \
OPTION( transitive, 1, 0, 1, "transitive reduction of binary clauses", 0) \
OPTION( transitiveeffort, 20, 0, 2e3, "effort in per mille", 0) \
OPTION( transitivekeep, 1, 0, 1, "keep transitivity candidates", 0) \
OPTION( tumble, 1, 0, 1, "tumbled external indices order", 0) \
NQTOPT( verbose, 0, 0, 3, "verbosity level", 0) \
OPTION( vivify, 1, 0, 1, "vivify clauses", 0) \
OPTION( vivifyeffort, 100, 0, 1e3, "effort in per mille", 0) \
OPTION( vivifyimply, 2, 0, 2, "remove implied redundant clauses", 0) \
OPTION( vivifyirred, 1, 1, 100, "relative irredundant effort", 0) \
OPTION( vivifykeep, 0, 0, 1, "keep vivification candidates", 0) \
OPTION( vivifytier1, 3, 1, 100, "relative tier1 effort", 0) \
OPTION( vivifytier2, 6, 1, 100, "relative tier2 effort", 0) \
OPTION( walkeffort, 5, 0, 1e6, "effort in per mille", 0) \
OPTION( walkfit, 1, 0, 1, "fit CB value to average clause length", 0) \
OPTION( walkinitially, 0, 0, 1, "initial local search", 0) \
OPTION( walkreuse, 1, 0, 2, "reuse walking results (2=always)", 0) \
OPTION( walkweighted, 1, 0, 1, "use clause weights", 0) \
OPTION( xors, 1, 0, 1, "extract and eliminate XOR gates", 0) \
OPTION( xorsbound, 1 ,0 , 1<<13, "minimum elimination bound", 0) \
OPTION( xorsclslim, 5, 3, 31, "XOR extraction clause size limit", 0) \

// *INDENT-OFF*

#define TARGET_SAT 2
#define TARGET_DEFAULT 1

#define STABLE_DEFAULT 1
#define STABLE_UNSAT 0

#define RESTARTINT_DEFAULT 1
#define RESTARTINT_SAT 50

#ifdef SAT
#undef TARGET_DEFAULT
#define TARGET_DEFAULT TARGET_SAT
#undef RESTARTINT_DEFAULT
#define RESTARTINT_DEFAULT RESTARTINT_SAT
#endif

#ifdef UNSAT
#undef STABLE_DEFAULT
#define STABLE_DEFAULT STABLE_UNSAT
#endif

#if defined(LOGGING) && !defined(QUIET)
#define LOGOPT OPTION
#else
#define LOGOPT(...) /**/
#endif

#ifndef QUIET
#define NQTOPT OPTION
#else
#define NQTOPT(...) /**/
#endif

#ifndef NDEBUG
#define DBGOPT OPTION
#else
#define DBGOPT(...) /**/
#endif

#ifdef EMBEDDED
#define EMBOPT OPTION
#else
#define EMBOPT(...) /**/
#endif

// *INDENT-ON*

typedef struct opt opt;

struct opt {
  const char *name;
#ifndef NOPTIONS
  int value;
  const int low;
  const int high;
#else
  const int value;
#endif
  const char *description;
  bool fixed;
};

extern const opt *kissat_options_begin;
extern const opt *kissat_options_end;

#define all_options(O) \
  opt const * O = kissat_options_begin; O != kissat_options_end; ++O

const char *kissat_parse_option_name(const char *arg, const char *name);
bool kissat_parse_option_value(const char *val_str, int *res_ptr);

#ifndef NOPTIONS

struct kissat;

void kissat_options_usage(struct kissat *);

const opt *kissat_options_has(const char *name);

#define kissat_options_max_name_buffer_size ((size_t) 32)

bool kissat_options_parse_arg(const char *arg, char *name, int *val_str);
void kissat_options_print_value(int value, char *buffer);

typedef struct options options;

struct options {
#define OPTION(N,V,L,H,D,F) int N;
  OPTIONS
#undef OPTION
};

void kissat_init_options(options *);

int kissat_options_get(const options *, const char *name);
int kissat_options_set_opt(options *, const opt *, int new_value, bool fixed);
int kissat_options_set(options *, const char *name, int new_value, bool fixed);

void kissat_print_embedded_option_list(void);
void kissat_print_option_range_list(void);

static inline int *kissat_options_ref(const options *options, const opt *o) {
  if (!o) {
    return 0;
  }
  assert(kissat_options_begin <= o);
  assert(o < kissat_options_end);
  return (int *) options + (o - kissat_options_begin);
}

#define GET_OPTION(NAME) ((int) solver->options.NAME)

#else

void kissat_init_options(void);
int kissat_options_get(const char *name);

#define GET_OPTION(N) kissat_options_ ## N

#define OPTION(N,V,L,H,D,F) static const int GET_OPTION(N) = (int)(V);
OPTIONS
#undef OPTION
#endif
#define GET1K_OPTION(NAME) (((int64_t) 1000) * GET_OPTION (NAME))
#endif
