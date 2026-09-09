#ifndef _promote_h_INCLUDED
#define _promote_h_INCLUDED

struct ring;
struct watcher;

unsigned recompute_glue (struct ring *, struct watcher *);
void promote_watcher (struct ring *, struct watcher *, unsigned new_glue);

#endif
