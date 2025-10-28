import sys
import pandas as pd


def process_file(filename, outputfile):
    df = pd.read_csv(filename)
    # df = df.iloc[:, [0,2, 6, 15, 24, 25, 26, 27, 28, 29, 30, -7, -6, -5]]

    col_list = [
        "OP CODE",
        "CORE COUNT",
        "DEVICE KERNEL DURATION [ns]",
        "PM IDEAL [ns]",
        "PM COMPUTE [ns]",
        "PM BANDWIDTH [ns]",
        "INPUT_0_W_PAD[LOGICAL]",
        "INPUT_0_Z_PAD[LOGICAL]",
        "INPUT_0_Y_PAD[LOGICAL]",
        "INPUT_0_X_PAD[LOGICAL]",
        "INPUT_0_LAYOUT",
        "INPUT_0_DATATYPE",
        "INPUT_0_MEMORY",
        "OUTPUT_0_W_PAD[LOGICAL]",
        "OUTPUT_0_Z_PAD[LOGICAL]",
        "OUTPUT_0_Y_PAD[LOGICAL]",
        "OUTPUT_0_X_PAD[LOGICAL]",
        "OUTPUT_0_LAYOUT",
        "OUTPUT_0_DATATYPE",
        "OUTPUT_0_MEMORY",
    ]
    df = df[col_list]

    df.rename(columns=lambda x: x.replace("INPUT_", "IN_"), inplace=True)
    df.rename(columns=lambda x: x.replace("OUTPUT_", "OUT_"), inplace=True)

    """
    PM IDEAL [ns] PM COMPUTE [ns]  PM BANDWIDTH [ns]a
    OUTPUT_0_W	OUTPUT_0_Z	OUTPUT_0_Y	OUTPUT_0_X	OUTPUT_0_LAYOUT	OUTPUT_0_DATATYPE	OUTPUT_0_MEMORY
    # Select single column
    name_column = df['Name']

    # Select multiple columns
    selected_columns = df[['Name', 'City']]
    """

    # first non-nan
    first_index = df["CORE COUNT"].first_valid_index()
    df = df.iloc[first_index:]

    # df = df.replace('','', regex=True)
    df = df.replace("InterleavedToShardedDeviceOperation", "I2S", regex=True)
    df = df.replace("ShardedToInterleavedDeviceOperation", "S2I", regex=True)
    df = df.replace("DeviceOperation", "", regex=True)
    df = df.replace("DEV_0_", "", regex=True)

    df = df.convert_dtypes()
    Total = df["DEVICE KERNEL DURATION [ns]"].sum()

    # Step 2: Basic analysis
    """
    print("\nSummary Statistics:")
    print(df.describe().applymap(lambda x: f'{x:.0f}') )
    print("----------")
    """
    df_sorted = df.sort_values(by="DEVICE KERNEL DURATION [ns]", ascending=False)

    df["DEVICE KERNEL DURATION [ns]"] = df.apply(lambda x: "{:,}".format(x["DEVICE KERNEL DURATION [ns]"]), axis=1)
    df.to_excel(outputfile + ".xlsx", index=False)
    print("Dataframe Head:")
    print(df.head(60))
    print("------------")
    print("Top runtime contributors:")
    df_sorted["DEVICE KERNEL DURATION [ns]"] = df_sorted.apply(
        lambda x: "{:,}".format(x["DEVICE KERNEL DURATION [ns]"]), axis=1
    )
    print(df_sorted.head(10))
    print("------------")
    print("------------")
    formatted_number = "{:,}".format(Total)
    print("Total nsec =", formatted_number)
    print("------------")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a filename as an argument.")
    else:
        filename = sys.argv[1]
        outputfile = sys.argv[2]
        process_file(filename, outputfile)
