import csv

import matplotlib.pyplot as plt


def load_data(file_name):
    data = []
    with open(file_name) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            data.append(row)
    return data


def plot(data, y_axis="", x_axis="", other_axes: list[str] = None):
    assert len(other_axes) == 2, "only works for fixing 2 other axes now..."
    choices = []
    for axis in other_axes:
        v = [int(d[axis]) for d in data]
        choices.append(sorted(set(v)))

    # TODO expand to more than 2 other_axes?
    for c in [[z0, z1] for z0 in choices[0] for z1 in choices[1]]:  # TODO change values for z1
        label = f"{other_axes[0]}={c[0]}, {other_axes[1]}={c[1]}"
        x = []
        y = []
        for p in data:
            if p[other_axes[0]] == str(c[0]) and p[other_axes[1]] == str(c[1]):
                x.append(int(p[x_axis]))
                y.append(float(p[y_axis]))
        plt.plot(x, y, "o-", label=label)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    raw_data = load_data("/Users/gfeng/height_sharded3.csv")
    for num_cores in [10]:  # TODO change this value
        filtered_data = [p for p in raw_data if int(p["num_cores"]) == num_cores]
        # plot(filtered_data, y_axis="duration", x_axis="m_size", other_axes=["k_size", "n_size"])
        plot(filtered_data, y_axis="duration", x_axis="n_size", other_axes=["k_size", "m_size"])
