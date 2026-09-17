#ifndef _value_h_INCLUDED
#define _value_h_INCLUDED

typedef signed char value;
typedef signed char mark;

#define VALUE(LIT) \
  (solver->values[assert ((LIT) < LITS), (LIT)])

#define MARK(LIT) \
  (solver->marks[assert ((LIT) < LITS), (LIT)])

#define BOOL_TO_VALUE(B) ((signed char)((B) ? -1 : 1))

#endif
