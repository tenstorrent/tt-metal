#ifndef QUIET

#include "statistics.h"
#include "message.h"
#include "ruler.h"
#include "tiers.h"
#include "utilities.h"

#include <inttypes.h>

void print_ring_statistics (struct ring *ring) {
  print_ring_profiles (ring);
  double search = ring->profiles.search.time;
  double walk = ring->profiles.solve.time;
  struct ring_statistics *s = &ring->statistics;
  struct context *c = s->contexts + SEARCH_CONTEXT;
  uint64_t conflicts = c->conflicts;
  uint64_t chronological = c->chronological;
  uint64_t decisions = c->decisions;
  uint64_t propagations = c->propagations;
  uint64_t jumped = c->jumped;
#ifdef METRICS
  uint64_t visits = 0;
  for (unsigned i = 0; i != SIZE_VISITS; i++)
    visits += c->visits[i];
#endif
  unsigned variables = ring->ruler->size;
  PRINTLN ("%-22s %17" PRIu64 " %13.2f per second", "conflicts:", conflicts,
           average (conflicts, search));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% conflicts",
           "chronological:", chronological,
           percent (chronological, conflicts));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f per conflict",
           "decisions:", decisions, average (decisions, conflicts));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% decisions",
           "  heap-decisions:", s->decisions.heap,
           percent (s->decisions.heap, decisions));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% decisions",
           "  negative-decisions:", s->decisions.negative,
           percent (s->decisions.negative, decisions));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% decisions",
           "  positive-decisions:", s->decisions.positive,
           percent (s->decisions.positive, decisions));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% decisions",
           "  queue-decisions:", s->decisions.queue,
           percent (s->decisions.queue, decisions));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% decisions",
           "  random-decisions:", s->decisions.random,
           percent (s->decisions.random, decisions));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f decisions",
           "  random-sequences:", s->random_sequences,
           average (s->decisions.random, s->random_sequences));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% learned",
           "eagerly-subsumed:", s->eagerly_subsumed,
           percent (s->eagerly_subsumed, s->learned.clauses));
  PRINTLN ("%-22s %17u %13.2f %% variables", "solving-fixed:", s->fixed,
           percent (s->fixed, variables));
  PRINTLN ("%-22s %17u %13.2f %% variables", "failed-literals:", s->failed,
           percent (s->failed, variables));
  PRINTLN ("%-22s %17u %13.2f %% variables", "lifted-literals:", s->lifted,
           percent (s->lifted, variables));
  PRINTLN ("%-22s %17u %13.2f %% variables", "fixed-variables:", s->fixed,
           percent (s->fixed, variables));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% fixed",
           "  learned-units:", s->learned.units,
           percent (s->learned.units, s->fixed));
  if (ring->pool) {
    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% fixed",
             "  imported-units:", s->imported.units,
             percent (s->imported.units, s->fixed));
    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% fixed",
             "  exported-units:", s->exported.units,
             percent (s->exported.units, s->fixed));
  }

  PRINTLN ("%-22s %17" PRIu64 " %13.2f thousands per second",
           "flips:", s->flips, average (s->flips, 1e3 * walk));

  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% per tried clause",
           "vivified-clauses:", s->vivify.succeeded,
           percent (s->vivify.succeeded, s->vivify.tried));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% per learned clause",
           "  vivify-tried:", s->vivify.tried,
           percent (s->vivify.tried, s->learned.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% per vivify-tried",
           "  vivify-reused:", s->vivify.reused,
           percent (s->vivify.reused, s->vivify.tried));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% vivified",
           "  vivify-strengthened:", s->vivify.strengthened,
           percent (s->vivify.strengthened, s->vivify.succeeded));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% vivified",
           "  vivify-subsumed:", s->vivify.subsumed,
           percent (s->vivify.subsumed, s->vivify.succeeded));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% vivified",
           "  vivify-implied:", s->vivify.implied,
           percent (s->vivify.implied, s->vivify.succeeded));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% strengthened",
           "  vivify-units:", s->vivify.units,
           percent (s->vivify.units, s->vivify.strengthened));

  PRINTLN ("%-22s %17" PRIu64 " %13.2f per learned clause",
           "learned-literals:", s->literals.learned,
           average (s->literals.learned, s->learned.clauses));
#ifdef METRICS
  PRINTLN ("%-22s %17" PRIu64 " %13.2f times learned literals",
           "  deduced-literals:", s->literals.deduced,
           average (s->literals.deduced, s->literals.learned));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% per deduced literal",
           "  minimized-literals:", s->literals.minimized,
           percent (s->literals.minimized, s->literals.deduced));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% per deduced literal",
           "  shrunken-literals:", s->literals.shrunken,
           percent (s->literals.shrunken, s->literals.deduced));
#endif

#ifdef METRICS
#define PRINT_CLAUSE_METRICS(NAME) \
  INSTANTIATE (1, SIZE_GLUE_STATISTICS - 1, NAME)
#else
#define PRINT_CLAUSE_METRICS(NAME) /**/
#endif
#define PRINT_CLAUSE_STATISTICS(NAME) \
  do { \
    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% " #NAME " clauses", \
             "  " #NAME "-binaries:", s->NAME.binaries, \
             percent (s->NAME.binaries, s->NAME.clauses)); \
    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% " #NAME " clauses", \
             "  " #NAME "-tier1:", s->NAME.tier1, \
             percent (s->NAME.tier1, s->NAME.clauses)); \
    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% " #NAME " clauses", \
             "  " #NAME "-tier2:", s->NAME.tier2, \
             percent (s->NAME.tier2, s->NAME.clauses)); \
    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% " #NAME " clauses", \
             "  " #NAME "-tier3:", s->NAME.tier3, \
             percent (s->NAME.tier3, s->NAME.clauses)); \
    PRINT_CLAUSE_METRICS (NAME); \
  } while (0)
#define MACRO(SIZE, NAME) \
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% " #NAME " clauses", \
           "  " #NAME "-glue" #SIZE ":", s->NAME.glue[SIZE], \
           percent (s->NAME.glue[SIZE], s->NAME.clauses))
  PRINTLN ("%-22s %17" PRIu64 " %13.2f per second",
           "learned-clauses:", s->learned.clauses,
           average (s->learned.clauses, search));
  PRINT_CLAUSE_STATISTICS (learned);
#ifdef METRICS
  uint64_t learned_glue_small = 0;
  for (unsigned glue = 1; glue != SIZE_GLUE_STATISTICS; glue++)
    learned_glue_small += s->learned.glue[glue];
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% learned clauses",
           "  learned-glue-small:", learned_glue_small,
           percent (learned_glue_small, s->learned.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% learned clauses",
           "  learned-glue-large:", s->learned.glue[0],
           percent (s->learned.glue[0], s->learned.clauses));
#endif

  PRINTLN ("%-22s %17" PRIu64 " %13.2f per learned",
           "bumped-clauses:", s->bumped,
           average (s->bumped, s->learned.clauses));
  print_tiers_bumped_statistics (ring);

  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% bumped",
           "promoted-clauses:", s->promoted.clauses,
           percent (s->promoted.clauses, s->bumped));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% promoted",
           "  promoted-kept1:", s->promoted.kept1,
           percent (s->promoted.kept1, s->promoted.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% promoted",
           "  promoted-kept2:", s->promoted.kept2,
           percent (s->promoted.kept2, s->promoted.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% promoted",
           "  promoted-kept3:", s->promoted.kept3,
           percent (s->promoted.kept3, s->promoted.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% promoted",
           "  promoted-tier1:", s->promoted.tier1,
           percent (s->promoted.tier1, s->promoted.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% promoted",
           "  promoted-tier2:", s->promoted.tier2,
           percent (s->promoted.tier2, s->promoted.clauses));

  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% propagations", "jumped:", jumped,
           percent (jumped, propagations));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f millions per second",
           "propagations:", propagations,
           average (propagations, 1e6 * search));
#ifdef METRICS
  PRINTLN ("%-22s %17" PRIu64 " %13.2f per propagation", "visits:", visits,
           average (visits, propagations));
#undef MACRO
#define MACRO(SIZE, DUMMY) \
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% visits", "  visits" #SIZE ":", \
           c->visits[SIZE], percent (c->visits[SIZE], visits))
  INSTANTIATE (SIZE_WATCHER_LITERALS + 1, SIZE_VISITS - 1);
#undef MACRO
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% visits",
           "  visits-large:", c->visits[0], percent (c->visits[0], visits));
#endif

  PRINTLN ("%-22s %17" PRIu64 " %13.2f conflict interval",
           "probings:", s->probings, average (conflicts, s->probings));

  PRINTLN ("%-22s %17" PRIu64 " %13.2f conflict interval",
           "reductions:", s->reductions,
           average (conflicts, s->reductions));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% learned",
           "  reduced-clauses:", s->reduced.clauses,
           percent (s->reduced.clauses, s->learned.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% reduced",
           "  reduced-tier1:", s->reduced.tier1,
           percent (s->reduced.tier1, s->reduced.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% reduced",
           "  reduced-tier2:", s->reduced.tier2,
           percent (s->reduced.tier2, s->reduced.clauses));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f %% reduced",
           "  reduced-tier3:", s->reduced.tier3,
           percent (s->reduced.tier3, s->reduced.clauses));

  if (ring->pool) {
    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% learned clauses",
             "imported-clauses:", s->imported.clauses,
             percent (s->imported.clauses, s->learned.clauses));
    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% imported clauses",
             "  diverged-imports:", s->diverged,
             percent (s->diverged, s->imported.clauses));
    PRINT_CLAUSE_STATISTICS (imported);

    {
      uint64_t subsumed =
          s->subsumed.binary.succeeded + s->subsumed.large.succeeded;
      uint64_t checked =
          s->subsumed.binary.checked + s->subsumed.large.checked;
      PRINTLN ("%-22s %17" PRIu64 " %13.2f %% checked clauses",
               "subsumed-clauses:", subsumed, percent (subsumed, checked));
      PRINTLN ("%-22s %17" PRIu64 " %13.2f %% checked clauses",
               "  subsumed-binary:", s->subsumed.binary.succeeded,
               percent (s->subsumed.binary.succeeded,
                        s->subsumed.binary.checked));
      PRINTLN (
          "%-22s %17" PRIu64 " %13.2f %% checked clauses",
          "  subsumed-large:", s->subsumed.large.succeeded,
          percent (s->subsumed.large.succeeded, s->subsumed.large.checked));
    }

    PRINTLN ("%-22s %17" PRIu64 " %13.2f %% learned clauses",
             "exported-clauses:", s->exported.clauses,
             percent (s->exported.clauses, s->learned.clauses));
    PRINT_CLAUSE_STATISTICS (exported);
  }

  PRINTLN ("%-22s %17" PRIu64 " %13.2f conflict interval",
           "rephased:", s->rephased, average (conflicts, s->rephased));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f conflict interval",
           "restarts:", s->restarts, average (conflicts, s->restarts));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f conflict interval",
           "simplifications:", s->simplifications,
           average (conflicts, s->simplifications));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f conflict interval",
           "switched:", s->switched, average (conflicts, s->switched));
  PRINTLN ("%-22s %17" PRIu64 " %13.2f flips per walked",
           "walked:", s->walked, average (s->flips, s->walked));
  fflush (stdout);
}

void print_ruler_statistics (struct ruler *ruler) {
  if (verbosity < 0)
    return;

  for (all_rings (ring)) {
    print_ring_statistics (ring);
    printf ("c\n");
  }

  print_ruler_profiles (ruler);

  double process = process_time ();
  double total = current_time () - start_time;
  double memory = maximum_resident_set_size () / (double) (1 << 20);

  struct ruler_statistics *s = &ruler->statistics;

  unsigned variables = ruler->size;

  printf ("c %-22s %17u %13.2f %% variables\n",
          "eliminated:", s->eliminated, percent (s->eliminated, variables));
  printf ("c %-22s %17u %13.2f %% eliminated variables\n",
          "definitions:", s->definitions,
          percent (s->definitions, s->eliminated));
  printf ("c %-22s %17" PRIu64 " %13.2f %% variables\n",
          "substituted:", s->substituted,
          percent (s->substituted, variables));
  printf ("c %-22s %17" PRIu64 " %13.2f %% subsumed clauses\n",
          "deduplicated:", s->deduplicated,
          percent (s->deduplicated, s->subsumed));
  printf ("c %-22s %17" PRIu64 " %13.2f %% subsumed clauses\n",
          "self-subsumed:", s->selfsubsumed,
          percent (s->selfsubsumed, s->subsumed));
  printf ("c %-22s %17" PRIu64 " %13.2f %% original clauses\n",
          "strengthened:", s->strengthened,
          percent (s->strengthened, s->original));
  printf ("c %-22s %17" PRIu64 "\n",
          "simplifications:", s->simplifications);
  printf ("c %-22s %17" PRIu64 " %13.2f %% original clauses\n",
          "subsumed:", s->subsumed, percent (s->subsumed, s->original));
  printf ("c %-22s %17zu %13.2f %% original clauses\n",
          "weakened:", s->weakened, percent (s->weakened, s->original));
  printf ("c %-22s %17u %13.2f %% total-fixed\n",
          "simplifying-fixed:", s->fixed.simplifying,
          percent (s->fixed.simplifying, s->fixed.total));
  printf ("c %-22s %17u %13.2f %% total-fixed\n",
          "solving-fixed:", s->fixed.solving,
          percent (s->fixed.solving, s->fixed.total));
  printf ("c %-22s %17u %13.2f %% variables\n",
          "total-fixed:", s->fixed.total,
          percent (s->fixed.total, variables));

  printf ("c\n");

  printf ("c %-30s %23.2f %%\n", "utilization:",
          percent (process / ruler->options.threads, total));
  printf ("c %-30s %23.2f seconds\n", "process-time:", process);
  printf ("c %-30s %23.2f seconds\n", "wall-clock-time:", total);
  printf ("c %-30s %23.2f MB\n", "maximum-resident-set-size:", memory);

  fflush (stdout);
}

#endif
