from datetime import datetime
from typing import Literal

from boltz.data.types import Record
from boltz.data.filter.dynamic.filter import DynamicFilter


class DateFilter(DynamicFilter):
    """A filter that filters complexes based on their date.

    The date can be the deposition, release, or revision date.
    If the date is not available, the previous date is used.

    If no date is available, the complex is rejected.

    """

    def __init__(
        self,
        date: str,
        ref: Literal["deposited", "revised", "released"],
    ) -> None:
        """Initialize the filter.

        Parameters
        ----------
        date : str, optional
            The maximum date of PDB entries to filter
        ref : Literal["deposited", "revised", "released"]
            The reference date to use.

        """
        self.filter_date = datetime.fromisoformat(date)
        self.ref = ref

        if ref not in ["deposited", "revised", "released"]:
            msg = (
                "Invalid reference date. Must be ",
                "deposited, revised, or released",
            )
            raise ValueError(msg)

    def filter(self, record: Record) -> bool:
        """Filter a record based on its date.

        Parameters
        ----------
        record : Record
            The record to filter.

        Returns
        -------
        bool
            Whether the record should be filtered.

        """
        structure = record.structure

        if self.ref == "deposited":
            date = structure.deposited
        elif self.ref == "released":
            date = structure.released
            if not date:
                date = structure.deposited
        elif self.ref == "revised":
            date = structure.revised
            if not date and structure.released:
                date = structure.released
            elif not date:
                date = structure.deposited

        if date is None or date == "":
            return False

        date = datetime.fromisoformat(date)
        return date <= self.filter_date
