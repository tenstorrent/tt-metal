#ifndef _compact_h_INCLUDED
#define _compact_h_INCLUDED

struct kissat;

unsigned kissat_compact_literals(struct kissat *, unsigned *mfixed_ptr);
void kissat_finalize_compacting(struct kissat *,
      unsigned vars, unsigned mfixed);
#endif
