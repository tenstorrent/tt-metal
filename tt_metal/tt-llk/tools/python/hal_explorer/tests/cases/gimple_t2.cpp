// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define N1 TTI_NOP
#define N2 TTI_NOP; TTI_NOP
#define N3 TTI_NOP; TTI_NOP; TTI_NOP
#define N4 TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP
// expected: a==0: 1 ; a==1: 2 ; a==2: 3; a>=3 ... chain
void chain(unsigned a) {
    if (a == 0) { N1; } else if (a == 1) { N2; } else if (a == 2) { N3; } else { N4; }
    TTI_SETDMAREG(0, 9, 0, 0);
}
// nested: a&&b -> 3 ; a&&!b -> 2 ; !a -> 1  (+1 tail)
void nested(unsigned a, unsigned b) {
    if (a) { N1; if (b) { N2; } else { N1; } }
    TTI_SETDMAREG(0, 9, 0, 0);
}
// switch fallthrough: 0->1+2+3=6 ; 1->2+3=5 ; 2->3 ; default->4
void sw(unsigned a) {
    switch (a) {
    case 0: N1; [[fallthrough]];
    case 1: N2; [[fallthrough]];
    case 2: N3; break;
    default: N4;
    }
    TTI_SETDMAREG(0, 9, 0, 0);
}
// early return: a-> 1 ; !a -> 1 + 2 = 3
void early(unsigned a) {
    N1;
    if (a) return;
    N2;
}
// goto: a -> N1 then skip N2 : 1+1 ; !a -> 1+2+1
void gt(unsigned a) {
    N1;
    if (a) goto done;
    N2;
done:
    TTI_SETDMAREG(0, 9, 0, 0);
}
// two-level early return
void early2(unsigned a, unsigned b) {
    if (a) { N1; return; }
    if (b) { N2; return; }
    N3;
}
