#include "allocate.h"
#include "analyze.h"
#include "backtrack.h"
#include "bump.h"
#include "decide.h"
#include "failed.h"
#include "learn.h"
#include "print.h"
#include "prophyper.h"
#include "proprobe.h"
#include "inline.h"
#include "sort.h"
#include "report.h"
#include "terminate.h"
#include "trail.h"

#include <inttypes.h>
#include <string.h>

static bool occurs_in_binary_clause(kissat *solver, const value *values,
      unsigned lit) {
  for (all_binary_blocking_watches(watch, WATCHES(lit))) {
    if (!watch.type.binary) {
      continue;
    }
    const unsigned other = watch.binary.lit;
    if (values[other]) {
      continue;
    }
    return true;
  }
  return false;
}

static size_t schedule_probes(kissat *solver, unsigned round,
      unsigneds *roots) {
  assert(!solver->level);
  assert(EMPTY_STACK(*roots));
  LOG("collecting roots of binary implication graph");
  assert(solver->watching);
  const value *const values = solver->values;
  value *marks = solver->marks;
  flags *flags = solver->flags;
  for (all_variables(idx)) {
    if (!flags[idx].active) {
      continue;
    }
    const unsigned lit = LIT(idx);
    const unsigned not_lit = NOT(lit);
    if (occurs_in_binary_clause(solver, values, lit)) {
      marks[lit] = 1;
    }
    if (occurs_in_binary_clause(solver, values, not_lit)) {
      marks[not_lit] = 1;
    }
  }
  for (all_clauses(c)) {
    if (c->garbage) {
      continue;
    }
    unsigned size = 0, first = INVALID_LIT, second = INVALID_LIT;
    for (all_literals_in_clause(lit, c)) {
      const value value = values[lit];
      if (value > 0) {
        kissat_mark_clause_as_garbage(solver, false, c);
        size = UINT_MAX;
        break;
      }
      if (value < 0) {
        continue;
      }
      if (!size++) {
        first = lit;
      } else if (size == 2) {
        second = lit;
      } else {
        break;
      }
    }
    if (size > 2) {
      continue;
    }
    assert(size == 2);
    assert(first != INVALID_LIT);
    assert(second != INVALID_LIT);
    marks[first] = 1;
    marks[second] = 1;
  }
  unsigned scheduled[2] = { 0, 0 };
  for (all_variables(idx)) {
    const unsigned lit = LIT(idx);
    const unsigned not_lit = NOT(lit);
    if (!flags[idx].active) {
      continue;
    }
    const bool pos = marks[lit];
    const bool neg = marks[not_lit];
    if (pos && neg) {
      LOG("skipping %s occurring positively and negatively",
            LOGVAR(idx));
      continue;
    }
    if (!pos && !neg) {
      LOG("skipping %s not occurring at all", LOGVAR(idx));
      continue;
    }
    unsigned root = INVALID_LIT;
    if (pos && !neg) {
      LOG("%s occurs only positively", LOGVAR(idx));
      root = not_lit;
    }
    if (!pos && neg) {
      LOG("%s occurs only negatively", LOGVAR(idx));
      root = lit;
    }
    LOG("collecting root %s", LOGLIT(root));
    PUSH_STACK(*roots, root);
    unsigned probe = flags[idx].probe;
    scheduled[probe]++;
  }
  memset(marks, 0, LITS * sizeof * marks);
  size_t res = SIZE_STACK(*roots);
  assert(res == scheduled[0] + scheduled[1]);
#ifdef QUIET
  (void) round;
#else
  kissat_phase(solver, "failed", GET(failed_computations),
        "round %u: scheduled %zu probes %.0f%% out of all variables",
        round, res, kissat_percent(res, solver->active));
  if (scheduled[1])
    kissat_phase(solver, "failed", GET(failed_computations),
          "round %u: prioritized %u probes %.0f%% of all scheduled",
          round, scheduled[1], kissat_percent(scheduled[1], res));
#endif
  return res;
}

static inline bool less_stable_probe(kissat *solver,
      const flags *const flags, const heap *scores,
      unsigned a, unsigned b) {
#ifdef NDEBUG
  (void) solver;
#endif
  const unsigned i = IDX(a);
  const unsigned j = IDX(b);
  const bool p = flags[i].probe;
  const bool q = flags[j].probe;
  if (!p && q) {
    return true;
  }
  if (p && !q) {
    return false;
  }
  const double s = kissat_get_heap_score(scores, i);
  const double t = kissat_get_heap_score(scores, j);
  if (s < t) {
    return true;
  }
  if (s > t) {
    return false;
  }
  return i < j;
}

static inline unsigned less_focused_probe(kissat *solver,
      const flags *const flags, const links *links,
      unsigned a, unsigned b) {
#ifdef NDEBUG
  (void) solver;
#endif
  const unsigned i = IDX(a);
  const unsigned j = IDX(b);
  const bool p = flags[i].probe;
  const bool q = flags[j].probe;
  if (!p && q) {
    return true;
  }
  if (p && !q) {
    return false;
  }
  const unsigned s = links[i].stamp;
  const unsigned t = links[j].stamp;
  return s < t;
}

#define LESS_STABLE_PROBE(A,B) \
  less_stable_probe (solver, flags, scores, (A), (B))

#define LESS_FOCUSED_PROBE(A,B) \
  less_focused_probe (solver, flags, links, (A), (B))

static void sort_stable_probes(kissat *solver, unsigneds *probes) {
  const flags *const flags = solver->flags;
  const heap *const scores = &solver->scores;
  SORT_STACK(unsigned, *probes, LESS_STABLE_PROBE);
}

static void sort_focused_probes(kissat *solver, unsigneds *probes) {
  const flags *const flags = solver->flags;
  const links *const links = solver->links;
  SORT_STACK(unsigned, *probes, LESS_FOCUSED_PROBE);
}

static void sort_probes(kissat *solver, unsigneds *probes) {
  if (solver->stable) {
    sort_stable_probes(solver, probes);
  } else {
    sort_focused_probes(solver, probes);
  }
}

static bool probe_round(kissat *solver, unsigned round,
      uint64_t ticks_limit, unsigneds *probes,
      unsigned *failed_ptr, uint64_t *resolved_ptr) {
#ifndef QUIET
  const unsigned scheduled = SIZE_STACK(*probes);
#endif
  unsigned *stamps = kissat_calloc(solver, LITS, sizeof * stamps);
  const value *const values = solver->values;
  flags *flags = solver->flags;

#if !defined(NDEBUG) && defined(METRICS)
  uint64_t resolved = solver->statistics.hyper_binary_resolved;
#endif
  uint64_t redundant = solver->statistics.clauses_redundant;
  uint64_t ticks_delta = ticks_limit - solver->statistics.probing_ticks;
  uint64_t hard_limit = solver->statistics.probing_ticks + 10 * ticks_delta;

  kissat_extremely_verbose(solver,
        "starting at %" PRIu64 " probing ticks "
        "limit %" PRIu64,
        solver->statistics.probing_ticks, ticks_limit);
  unsigned probed = 0;
  unsigned failed = 0;
#ifndef QUIET
  unsigned continued = 0;

  int verbosity = kissat_verbosity(solver);
  const uint64_t upper_report_probed_delta = (scheduled + 19) / 20;
  uint64_t report_probed_limit = 1, report_probed_delta = 1;
#endif

  while (!EMPTY_STACK(*probes)) {
    const unsigned probe = POP_STACK(*probes);
    if (values[probe]) {
      continue;
    }
    const unsigned probe_idx = IDX(probe);
    flags[probe_idx].probe = false;
    if (failed && stamps[probe] == failed) {
      continue;
    }
    if (solver->stable)
      LOG("probing %s[%g]", LOGLIT(probe),
            kissat_get_heap_score(&solver->scores, IDX(probe)));
    else {
      LOG("probing %s{%u}", LOGLIT(probe), LINK(IDX(probe)).stamp);
    }
    probed++;
    unsigned *saved = solver->propagate;
    assert(kissat_propagated(solver));
    kissat_internal_assume(solver, probe);
    assert(solver->level == 1);
    assert(values[probe] > 0);
    clause *conflict = kissat_hyper_propagate(solver, 0);
    if (conflict) {
      LOG("failed literal %s", LOGLIT(probe));
      INC(failed_units);
      failed++;
      (void) kissat_analyze(solver, conflict);
      assert(!solver->level);
      assert(values[probe] < 0);
      if (kissat_probing_propagate(solver, 0, true)) {
        break;
      }
    } else {
      assert(solver->level == 1);
      assert(kissat_propagated(solver));
      const unsigned *const propagated = solver->propagate;
      while (saved != propagated) {
        stamps[*saved++] = failed;
      }
      kissat_backtrack_without_updating_phases(solver, 0);
    }

    bool terminate = false;
    if (TERMINATED(failed_terminated_1)) {
      LOG("terminating");
      terminate = true;
    } else if (solver->statistics.probing_ticks > ticks_limit) {
      const double fraction = GET_OPTION(failedcont) * 1e-2;
      const unsigned lower_failed_limit = fraction * probed;

      if (solver->statistics.probing_ticks > hard_limit) {
        kissat_extremely_verbose(solver,
              "hard ticks limit %" PRIu64
              " hit after %" PRIu64,
              hard_limit,
              solver->statistics.probing_ticks);
        terminate = true;
      } else if (lower_failed_limit > failed) {
#ifndef QUIET
        if (continued)
          kissat_extremely_verbose(solver,
                "ticks limit %" PRIu64
                " hit after %" PRIu64
                " before hard limit while "
                "continuing %u times",
                ticks_limit,
                solver->statistics.probing_ticks,
                continued);
        else
          kissat_extremely_verbose(solver,
                "ticks limit %" PRIu64
                " hit after %" PRIu64
                " without continuing",
                ticks_limit,
                solver->statistics.probing_ticks);
#endif
        terminate = true;
      } else {
        LOG("continue probing since "
              "%u failed >= limit %u = %.g * probed %u",
              failed, lower_failed_limit, fraction, probed);
#ifndef QUIET
        continued++;
#endif
      }
    }
#ifndef QUIET
    if (verbosity >= 0 &&
          (probed >= report_probed_limit || EMPTY_STACK(*probes))) {
      uint64_t resolvents =
            solver->statistics.clauses_redundant - redundant;
      kissat_extremely_verbose(solver,
            "%u probed %.0f%%, "
            "%u failed %.0f%%, "
            "%u continued %.0f%%, "
            "%" PRIu64 " resolvents %.2f per probe",
            probed,
            kissat_percent(probed, scheduled),
            failed,
            kissat_percent(failed, probed),
            continued,
            kissat_percent(continued, probed),
            resolvents,
            kissat_average(resolvents, probed));
      report_probed_delta *= 10;
      if (report_probed_delta > upper_report_probed_delta) {
        report_probed_delta = upper_report_probed_delta;
      }

      report_probed_limit += report_probed_delta;
    }
#endif
    if (!terminate) {
      continue;
    }
#ifndef QUIET
    unsigned remain = SIZE_STACK(*probes);
#endif
    unsigned prioritized = 0;
    for (all_stack(unsigned, lit, *probes))
      if (flags[IDX(lit)].probe) {
        prioritized++;
      }

    if (!prioritized) {
      kissat_phase(solver, "failed",
            GET(failed_computations),
            "round %u: prioritizing remaining %u probes",
            round, remain);
      while (!EMPTY_STACK(*probes)) {
        const unsigned other = POP_STACK(*probes);
        const unsigned other_idx = IDX(other);
        assert(!flags[other_idx].probe);
        flags[other_idx].probe = true;
      }
    } else
      kissat_phase(solver, "failed",
            GET(failed_computations),
            "round %u: keeping %u probes %.0f%% prioritized",
            round, prioritized,
            kissat_percent(prioritized, remain));
    break;
  }
  kissat_dealloc(solver, stamps, LITS, sizeof * stamps);

  redundant = solver->statistics.clauses_redundant - redundant;
#if !defined(NDEBUG) && defined(METRICS)
  assert(resolved <= solver->statistics.hyper_binary_resolved);
  resolved = solver->statistics.hyper_binary_resolved - resolved;
  assert(redundant == resolved);
#endif

#ifndef QUIET
  if (continued)
    kissat_very_verbose(solver,
          "continued %u times %.0f%% even though limit hit",
          continued, kissat_percent(continued, probed));

  kissat_phase(solver, "failed",
        GET(failed_computations),
        "round %u: %u failed (%.0f%%) "
        "of %u probed (%.0f%% scheduled)",
        round,
        failed, kissat_percent(failed, probed),
        probed, kissat_percent(probed, scheduled));

  kissat_phase(solver, "failed", GET(failed_computations),
        "round %u: %" PRIu64 " hyper binary resolvents "
        "%.1f per probe", round, redundant,
        kissat_average(redundant, probed));
#else
  (void) round;
#endif
  *resolved_ptr += redundant;
  *failed_ptr += failed;

  if (solver->inconsistent) {
    kissat_phase(solver, "failed",
          GET(failed_computations),
          "last round produced empty clause");
    return false;
  }

  if (!failed)
    kissat_phase(solver, "failed",
          GET(failed_computations),
          "no failed literal produced in last round");
  if (!redundant)
    kissat_phase(solver, "failed",
          GET(failed_computations),
          "no hyper binary resolvent produced in last round");
#ifndef QUIET
  const bool success = failed || redundant;
  REPORT(!success, 'f');
#endif
  return failed;
}

void kissat_failed_literal_computation(kissat *solver) {
  if (solver->inconsistent) {
    return;
  }

  assert(!solver->level);
  assert(solver->probing);
  assert(solver->watching);

  if (TERMINATED(failed_terminated_2)) {
    return;
  }
  if (!GET_OPTION(failed)) {
    return;
  }

  RETURN_IF_DELAYED(failed);

  START(failed);
  INC(failed_computations);
#if !defined(NDEBUG) || defined(METRICS)
  assert(!solver->failed_probing);
  solver->failed_probing = true;
#endif

  assert(kissat_propagated(solver));
  assert(kissat_trail_flushed(solver));

  unsigneds roots;
  INIT_STACK(roots);

  SET_EFFORT_LIMIT(ticks_limit, failed, probing_ticks,
        kissat_linear(1 + solver->active));

  const unsigned max_rounds = GET_OPTION(failedrounds);
  unsigned round = 1;

  uint64_t resolved = 0;
  unsigned failed = 0;

  for (;;) {
    size_t scheduled = schedule_probes(solver, round, &roots);
    if (!scheduled) {
      kissat_phase(solver, "failed",
            GET(failed_computations),
            "no roots found in binary implication graph");
      break;
    }
    sort_probes(solver, &roots);
    if (!probe_round(solver, round, ticks_limit,
                &roots, &failed, &resolved)) {
      break;
    }
    if (!EMPTY_STACK(roots)) {
#ifndef QUIET
      size_t remain = SIZE_STACK(roots);
      kissat_phase(solver, "failed",
            GET(failed_computations),
            "incomplete probing with %zu remaining %.0f%%",
            remain, kissat_percent(remain, scheduled));
#endif
      break;
    }
    if (solver->statistics.probing_ticks > ticks_limit) {
      kissat_phase(solver, "failed",
            GET(failed_computations),
            "no more rounds since ticks limit hit");
      break;
    }
    if (round == max_rounds) {
      kissat_phase(solver, "failed",
            GET(failed_computations),
            "completed probing until maximum round");
      break;
    }
    round++;
  }
  RELEASE_STACK(roots);

  if (failed)
    kissat_phase(solver, "failed", GET(failed_computations),
          "%u failed literals in total in %u rounds", failed, round);
  else
    kissat_phase(solver, "failed", GET(failed_computations),
          "no failed literals in %u rounds", round);

  if (resolved)
    kissat_phase(solver, "failed", GET(failed_computations),
          "%" PRIu64 " hyper binary resolvents in total in %u rounds",
          resolved, round);
  else
    kissat_phase(solver, "failed", GET(failed_computations),
          "no hyper binary resolvents in total in %u rounds", round);

  bool success = failed + resolved;
  UPDATE_DELAY(success, failed);
#if !defined(NDEBUG) || defined(METRICS)
  assert(solver->failed_probing);
  solver->failed_probing = false;
#endif
  STOP(failed);
}
