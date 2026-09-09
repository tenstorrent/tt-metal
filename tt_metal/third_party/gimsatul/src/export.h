#ifndef _export_h_INCLUDED
#define _export_h_INCLUDED

struct clause;
struct ring;
struct watch;

void export_units (struct ring *);
void export_binary_clause (struct ring *, struct watch *);
void export_large_clause (struct ring *, struct clause *);
void flush_pool (struct ring *);

#endif
