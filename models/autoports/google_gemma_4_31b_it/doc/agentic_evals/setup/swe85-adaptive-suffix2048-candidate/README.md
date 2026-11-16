# Adaptive suffix 2048 candidate

Candidate only: not admitted or run. Five unresolved native motifs are retained for both served and full modes: Requests5414 calls 10 and 11, Matplotlib20488 call86, Requests/call91 canonical prefix, and Sphinx10323 call70. Each input uses the exact last 2048 original prompt IDs, preserves its original chosen continuation (64 positions except Sphinx call70’s original 37), and keeps temperature 1.0, top_p 0.95, top_k 20. Each mode runs the chosen branch plus 512 free tokens for seeds 42–44.

The point forecast preserves each suffix1024 trial’s observed stop length. Since the longer suffix changes the trajectory, the upper range budgets all free runs to the full 512-token cap (511 replay calls) before applying the 1.7× variable-work multiplier; the measured mode load stays fixed. Current point/range estimates are served 358.89s / 293.66–640.62s and full 486.14s / 363.65–1062.06s. CPU-only validation passed; no device was used.
