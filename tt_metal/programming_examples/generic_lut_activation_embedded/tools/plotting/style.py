"""Shared plotting style for generic_lut_activation_embedded experiment figures."""


def apply_tufte_style(plt, *, compact=False):
    font_size = 8 if compact else 9
    label_size = 9 if compact else 10
    tick_size = 8 if compact else 9
    legend_size = 8 if compact else 9

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = label_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size
    plt.rcParams["legend.fontsize"] = legend_size
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.labelcolor"] = "#333333"
    plt.rcParams["xtick.color"] = "#333333"
    plt.rcParams["ytick.color"] = "#333333"
    plt.rcParams["text.color"] = "#333333"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams["grid.alpha"] = 0.12
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["agg.path.chunksize"] = 10000
