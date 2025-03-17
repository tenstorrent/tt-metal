import matplotlib.pyplot as plt
import numpy as np

# Problem sizes
problem_sizes = [
    32 * 32,
    3 * 32 * 32,
    3 * 128 * 128,
    3 * 512 * 512,
    3 * 1024 * 1024,
    3 * 2048 * 2048,
    3 * 4096 * 4096,
    3 * 8192 * 8192,
    3 * 16384 * 16384,
    3 * 22912 * 22912,
]

# Execution times (ms) for RTA
rta_gelu = [
    0.012667655945,
    0.012613296509,
    0.012639760971,
    0.022520303726,
    0.084073781967,
    0.330229043961,
    1.314888954163,
    5.253144979477,
    21.005931138992,
    41.078379869461,
]
rta_silu = [
    0.012448310852,
    0.012214422226,
    0.012541294098,
    0.023902893066,
    0.089510917664,
    0.351879358292,
    1.401370763779,
    5.599013328552,
    22.389104127884,
    43.783565759659,
]
rta_tanh = [
    0.012572288513,
    0.012039184570,
    0.012364625931,
    0.012359380722,
    0.026406526566,
    0.100387811661,
    0.396591186523,
    1.580932617188,
    6.318182945251,
    12.353938102722,
]

# Execution times (ms) for CTA
cta_gelu = [
    0.012231111526,
    0.012127399445,
    0.015724658966,
    0.069407224655,
    0.108220100403,
    0.356472015381,
    1.428643226624,
    5.372735738754,
    20.903466463089,
    40.616855621338,
]
cta_silu = [
    0.011836051941,
    0.011855840683,
    0.015942811966,
    0.046741485596,
    0.112960338593,
    0.382395267487,
    1.477602958679,
    5.665860652924,
    22.388941526413,
    43.530015945435,
]
cta_tanh = [
    0.011742353439,
    0.011813402176,
    0.015769481659,
    0.038436412811,
    0.087965488434,
    0.313952207565,
    1.258586168289,
    4.437422275543,
    16.420392513275,
    30.770388841629,
]

# Create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Titles and data for each plot
titles = ["GELU Execution Time (RTA vs CTA)", "SiLU Execution Time (RTA vs CTA)", "Tanh Execution Time (RTA vs CTA)"]
rta_data = [rta_gelu, rta_silu, rta_tanh]
cta_data = [cta_gelu, cta_silu, cta_tanh]

for ax, title, rta_times, cta_times in zip(axes, titles, rta_data, cta_data):
    ax.plot(problem_sizes, rta_times, marker="o", linestyle="-", label="RTA")
    ax.plot(problem_sizes, cta_times, marker="s", linestyle="-", label="CTA")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem Volume")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
# plt.show()
plt.savefig("rta_vs_cta.png")
