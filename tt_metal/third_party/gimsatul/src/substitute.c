#include "substitute.h"
#include "message.h"
#include "ruler.h"
#include "simplify.h"
#include "trace.h"
#include "utilities.h"

static unsigned *find_equivalent_literals (struct simplifier *simplifier,
                                           unsigned round) {
  struct ruler *ruler = simplifier->ruler;
  size_t bytes = 2 * ruler->compact * sizeof (unsigned);
  unsigned *marks = allocate_and_clear_block (bytes);
  unsigned *reaches = allocate_and_clear_block (bytes);
  unsigned *repr = allocate_block (bytes);
  for (all_ruler_literals (lit))
    repr[lit] = lit;
  struct unsigneds scc;
  struct unsigneds work;
  INIT (scc);
  INIT (work);
  bool *eliminated = simplifier->eliminated;
  signed char *values = (signed char *) ruler->values;
  unsigned marked = 0, equivalences = 0;
  for (all_ruler_literals (root)) {
    if (eliminated[IDX (root)])
      continue;
    if (values[root])
      continue;
    if (marks[root])
      continue;
    assert (EMPTY (scc));
    assert (EMPTY (work));
    assert (marked < UINT_MAX);
    PUSH (work, root);
    while (!EMPTY (work)) {
      unsigned lit = POP (work);
      if (lit == INVALID) {
        lit = POP (work);
        unsigned not_lit = NOT (lit);
        unsigned lit_reaches = reaches[lit];
        struct clauses *clauses = &OCCURRENCES (not_lit);
        for (all_clauses (clause, *clauses)) {
          if (!is_binary_pointer (clause))
            continue;
          unsigned other = other_pointer (clause);
          if (values[other])
            continue;
          if (eliminated[IDX (other)])
            continue;
          unsigned other_reaches = reaches[other];
          if (other_reaches < lit_reaches)
            lit_reaches = other_reaches;
        }
        reaches[lit] = lit_reaches;
        unsigned lit_mark = marks[lit];
        if (lit_reaches != lit_mark)
          continue;
        unsigned *end = scc.end, *p = end, other, new_repr = lit;
        while ((other = *--p) != lit)
          if (other < new_repr)
            new_repr = other;
        scc.end = p;
        while (p != end) {
          unsigned other = *p++;
          reaches[other] = INVALID;
          if (other == new_repr)
            continue;
          repr[other] = new_repr;
          equivalences++;
          ROG ("literal %s is equivalent to representative %s",
               ROGLIT (other), ROGLIT (new_repr));
          if (other == NOT (new_repr)) {
            very_verbose (0, "%s", "empty resolvent");
            trace_add_unit (&ruler->trace, other);
            assign_ruler_unit (ruler, other);
            trace_add_empty (&ruler->trace);
            ruler->inconsistent = true;
            goto DONE;
          }
        }
      } else {
        if (marks[lit])
          continue;
        assert (marked < UINT_MAX);
        reaches[lit] = marks[lit] = ++marked;
        PUSH (work, lit);
        PUSH (work, INVALID);
        PUSH (scc, lit);
        unsigned not_lit = NOT (lit);
        struct clauses *clauses = &OCCURRENCES (not_lit);
        for (all_clauses (clause, *clauses)) {
          if (!is_binary_pointer (clause))
            continue;
          unsigned other = other_pointer (clause);
          if (values[other])
            continue;
          if (eliminated[IDX (other)])
            continue;
          if (marks[other])
            continue;
          PUSH (work, other);
        }
      }
    }
  }
DONE:
  RELEASE (scc);
  RELEASE (work);
  free (reaches);
  free (marks);
  verbose (0, "[%u] found %u new equivalent literal pairs", round,
           equivalences);
  if (equivalences && !ruler->inconsistent)
    return repr;
  free (repr);
  return 0;
}

static void substitute_clause (struct simplifier *simplifier, unsigned src,
                               unsigned dst, struct clause *clause) {
  struct ruler *ruler = simplifier->ruler;
  ROGCLAUSE (clause, "substituting");
  signed char *values = (signed char *) ruler->values;
  signed char dst_value = values[dst];
  if (dst_value > 0) {
    ROG ("satisfied replacement literal %s", ROGLIT (dst));
    return;
  }
  struct unsigneds *resolvent = &simplifier->resolvent;
  CLEAR (*resolvent);
  unsigned not_dst = NOT (dst);
  if (is_binary_pointer (clause)) {
    assert (lit_pointer (clause) == src);
    unsigned other = other_pointer (clause);
    if (other == not_dst) {
      ROG ("resulting clause tautological since it "
           "contains both %s and %s",
           ROGLIT (dst), ROGLIT (other));
      return;
    }
    if (other != dst) {
      signed char other_value = values[other];
      if (other_value > 0) {
        ROG ("clause already satisfied by %s", ROGLIT (other));
        return;
      }
      if (!other_value)
        PUSH (*resolvent, other);
    }
  } else {
    assert (!clause->garbage);
    for (all_literals_in_clause (other, clause)) {
      if (other == src)
        continue;
      if (other == dst)
        continue;
      if (other == not_dst) {
        ROG ("resulting clause tautological since it "
             "contains both %s and %s",
             ROGLIT (dst), ROGLIT (other));
        return;
      }
      signed char other_value = values[other];
      if (other_value < 0)
        continue;
      if (other_value > 0) {
        ROG ("clause already satisfied by %s", ROGLIT (other));
        return;
      }
      PUSH (*resolvent, other);
    }
  }
  if (!dst_value)
    PUSH (*resolvent, dst);
  add_resolvent (simplifier);
}

static void substitute_literal (struct simplifier *simplifier, unsigned src,
                                unsigned dst) {
  struct ruler *ruler = simplifier->ruler;
  assert (!ruler->values[src]);
  ROG ("substituting literal %s with %s", ROGLIT (src), ROGLIT (dst));
  assert (!simplifier->eliminated[IDX (src)]);
  assert (!simplifier->eliminated[IDX (dst)]);
  assert (src != NOT (dst));
  assert (dst < src);
  struct clauses *clauses = &OCCURRENCES (src);
  for (all_clauses (clause, *clauses)) {
    if (!is_binary_pointer (clause) && clause->garbage)
      continue;
    substitute_clause (simplifier, src, dst, clause);
    if (ruler->inconsistent)
      break;
    recycle_clause (simplifier, clause, src);
  }
  RELEASE (*clauses);
  struct unsigneds *extension = &ruler->extension[0];
  ROGBINARY (NOT (src), dst,
             "pushing on extension stack with witness literal %s",
             ROGLIT (NOT (src)));
  unsigned *unmap = ruler->unmap;
  PUSH (*extension, INVALID);
  PUSH (*extension, unmap_literal (unmap, src));
  PUSH (*extension, unmap_literal (unmap, NOT (dst)));
  if (SGN (src)) {
    unsigned idx = IDX (src);
    ROG ("marking %s as aliminated", ROGVAR (idx));
    ruler->statistics.substituted++;
    assert (ruler->statistics.active);
    ruler->statistics.active--;
    assert (!simplifier->eliminated[idx]);
    simplifier->eliminated[idx] = 1;
  }
}

static unsigned
substitute_equivalent_literals (struct simplifier *simplifier,
                                unsigned *repr) {
  struct ruler *ruler = simplifier->ruler;

  unsigned other;
  if (ruler->options.proof.file)
    for (all_positive_ruler_literals (lit))
      if ((other = repr[lit]) != lit) {
        trace_add_binary (&ruler->trace, NOT (lit), other);
        trace_add_binary (&ruler->trace, lit, NOT (other));
      }

  unsigned substituted = 0;
  signed char *values = (signed char *) ruler->values;
  for (all_ruler_indices (idx)) {
    if (ruler->terminate)
      break;
    unsigned lit = LIT (idx);
    if (values[lit])
      continue;
    unsigned other = repr[lit];
    if (other == lit)
      continue;
    substitute_literal (simplifier, lit, other);
    substituted++;
    if (ruler->inconsistent)
      break;
    if (values[lit])
      continue;
    unsigned not_lit = NOT (lit);
    unsigned not_other = NOT (other);
    assert (repr[not_lit] == not_other);
    substitute_literal (simplifier, not_lit, not_other);
    if (ruler->inconsistent)
      break;
  }

  if (ruler->options.proof.file)
    for (all_positive_ruler_literals (lit))
      if ((other = repr[lit]) != lit) {
        trace_delete_binary (&ruler->trace, NOT (lit), other);
        trace_delete_binary (&ruler->trace, lit, NOT (other));
      }

  RELEASE (simplifier->resolvent);

  return substituted;
}

bool equivalent_literal_substitution (struct simplifier *simplifier,
                                      unsigned round) {
#ifndef QUIET
  struct ruler *ruler = simplifier->ruler;
  double substitution_start = START (ruler, substitute);
#endif
  message (0, 0);
  unsigned *repr = find_equivalent_literals (simplifier, round);
  unsigned substituted = 0;
  if (repr) {
    substituted = substitute_equivalent_literals (simplifier, repr);
    free (repr);
  }
#ifndef QUIET
  double substitution_end = STOP (ruler, substitute);
  message (0, "[%u] substituted %u variables %.0f%% in %.2f seconds", round,
           substituted, percent (substituted, ruler->size),
           substitution_end - substitution_start);
#endif
  return substituted;
}
