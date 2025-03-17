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
    2 * 22912 * 22912,
]

# Execution times (ms) for RTA
rta_gelu = [
    0.012418508530,
    0.012265682220,
    0.012321233749,
    0.019392013550,
    0.072453498840,
    0.284643888474,
    1.133425951004,
    4.528164148331,
    18.107079029083,
    35.410610675812,
]
rta_silu = [
    0.012409210205,
    0.012076616287,
    0.012536287308,
    0.020592689514,
    0.077223777771,
    0.303289175034,
    1.207948207855,
    4.826276063919,
    19.299471616745,
    37.741742849350,
]
rta_tanh = [
    0.012416839600,
    0.011887550354,
    0.012511968613,
    0.012321472168,
    0.022738695145,
    0.086794853210,
    0.341921091080,
    1.362710714340,
    5.446412801743,
    26.518051385880,
]

# Execution times (ms) for CTA
cta_gelu = [
    0.011939525604,
    0.012176036835,
    0.015496492386,
    0.039479494095,
    0.093182086945,
    0.307259559631,
    1.231035470963,
    4.631619930267,
    18.019266128540,
    35.010937213898,
]
cta_silu = [
    0.011464595795,
    0.011434078217,
    0.016192913055,
    0.040194749832,
    0.097850561142,
    0.327695369720,
    1.274075746536,
    4.884420633316,
    19.322363853455,
    37.522536516190,
]
cta_tanh = [
    0.011322259903,
    0.011646747589,
    0.014835357666,
    0.033137559891,
    0.076003313065,
    0.268887758255,
    1.074993848801,
    3.795267343521,
    14.103883981705,
    26.514275789261,
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
